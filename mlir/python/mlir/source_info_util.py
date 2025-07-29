# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from collections.abc import Iterator
import contextlib
import dataclasses
import functools
import itertools
import os.path
import re
import sysconfig
import threading
import types
from typing import NamedTuple, Optional
from .ir import Location

# import jax.version
# from jax._src.lib import xla_client

from . import traceback_util
from .traceback_util import (
    TracebackCaches,
    Traceback,
    _traceback_caches,
    _traceback_in_locations_limit,
    _include_full_tracebacks_in_locations,
)

traceback_util.register_exclusion(__file__)


@dataclasses.dataclass(frozen=True)
class Frame:
    file_name: str
    function_name: str
    start_line: int
    start_column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None


_exclude_paths: list[str] = [
    # Attach the separator to make sure that .../jax does not end up matching
    # .../jax_triton and other packages that might have a jax prefix.
    # os.path.dirname(jax.version.__file__) + os.sep,
    # Also exclude stdlib as user frames. In a non-standard Python runtime,
    # the following two may be different.
    sysconfig.get_path("stdlib"),
    os.path.dirname(sysconfig.__file__),
]


@functools.cache
def _exclude_path_regex() -> re.Pattern[str]:
    # The regex below would not handle an empty set of exclusions correctly.
    assert len(_exclude_paths) > 0
    return re.compile("|".join(f"^{re.escape(path)}" for path in _exclude_paths))


def register_exclusion(path: str):
    _exclude_paths.append(path)
    _exclude_path_regex.cache_clear()
    is_user_filename.cache_clear()


# Explicit inclusions take priority over exclude paths.
_include_paths: list[str] = []


@functools.cache
def _include_path_regex() -> re.Pattern[str]:
    patterns = [f"^{re.escape(path)}" for path in _include_paths]
    patterns.append("_test.py$")
    return re.compile("|".join(patterns))


def register_inclusion(path: str):
    _include_paths.append(path)
    _include_path_regex.cache_clear()
    is_user_filename.cache_clear()


class Scope(NamedTuple):
    name: str

    def wrap(self, stack: list[str]):
        stack.append(self.name)


class Transform(NamedTuple):
    name: str

    def wrap(self, stack: list[str]):
        if stack:
            stack[-1] = f"{self.name}({stack[-1]})"
        else:
            stack.append(f"{self.name}()")


@dataclasses.dataclass(frozen=True)
class NameStack:
    stack: tuple[Scope | Transform, ...] = ()

    def extend(self, name: str) -> NameStack:
        return NameStack((*self.stack, Scope(name)))

    def transform(self, transform_name: str) -> NameStack:
        return NameStack((*self.stack, Transform(transform_name)))

    def __getitem__(self, idx: slice) -> NameStack:
        return NameStack(self.stack[idx])

    def __len__(self):
        return len(self.stack)

    def __add__(self, other: NameStack) -> NameStack:
        return NameStack(self.stack + other.stack)

    def __radd__(self, other: NameStack) -> NameStack:
        return NameStack(other.stack + self.stack)

    def __str__(self) -> str:
        scope: list[str] = []
        for elem in self.stack[::-1]:
            elem.wrap(scope)
        return "/".join(reversed(scope))


def new_name_stack(name: str = "") -> NameStack:
    name_stack = NameStack()
    if name:
        name_stack = name_stack.extend(name)
    return name_stack


class SourceInfo:
    traceback: Traceback | None
    name_stack: NameStack

    # It's slightly faster to use a class with __slots__ than a NamedTuple.
    __slots__ = ["traceback", "name_stack"]

    def __init__(self, traceback: Traceback | None, name_stack: NameStack):
        self.traceback = traceback
        self.name_stack = name_stack

    def replace(
        self, *, traceback: Traceback | None = None, name_stack: NameStack | None = None
    ) -> SourceInfo:
        return SourceInfo(
            self.traceback if traceback is None else traceback,
            self.name_stack if name_stack is None else name_stack,
        )


def new_source_info() -> SourceInfo:
    return SourceInfo(None, NameStack())


@functools.cache
def is_user_filename(filename: str) -> bool:
    """Heuristic that guesses the identity of the user's code in a stack trace."""
    return (
        _include_path_regex().search(filename) is not None
        or _exclude_path_regex().search(filename) is None
    )


def raw_frame_to_frame(code: types.CodeType, lasti: int) -> Frame:
    if sys.version_info.minor >= 11:
        loc = Traceback.code_addr2location(code, lasti)
        start_line, start_column, end_line, end_column = loc
        frame = Frame(
            file_name=code.co_filename,
            function_name=code.co_qualname,
            start_line=start_line,
            start_column=start_column,
            end_line=end_line,
            end_column=end_column,
        )
    else:
        start_line = Traceback.code_addr2line(code, lasti)
        frame = Frame(
            file_name=code.co_filename,
            function_name=code.co_name,
            start_line=start_line,
            start_column=0,
        )

    return frame


def user_frames(traceback: Traceback | None) -> Iterator[Frame]:
    """Iterator over the user's frames, filtering jax-internal frames."""
    # Guess the user's frame is the innermost frame not in the jax source tree or
    # Python stdlib. We don't use traceback_util.path_starts_with because that
    # incurs filesystem access, which may be slow; we call this function when
    # e.g. adding source provenance annotations to XLA lowerings, so we don't
    # want to incur the cost. We consider files that end with _test.py as user
    # frames, to allow testing this mechanism from tests.
    code, lasti = traceback.raw_frames() if traceback else ([], [])
    return (
        raw_frame_to_frame(code[i], lasti[i])
        for i in range(len(code))
        if is_user_filename(code[i].co_filename)
    )


@functools.lru_cache(maxsize=64)
def user_frame(traceback: Traceback | None) -> Frame | None:
    return next(user_frames(traceback), None)


def _summarize_frame(frame: Frame) -> str:
    if frame.start_column != 0:
        return (
            f"{frame.file_name}:{frame.start_line}:{frame.start_column} "
            f"({frame.function_name})"
        )
    else:
        return f"{frame.file_name}:{frame.start_line} ({frame.function_name})"


def summarize(source_info: SourceInfo, num_frames=1) -> str:
    frames = itertools.islice(user_frames(source_info.traceback), num_frames)
    frame_strs = [_summarize_frame(frame) if frame else "unknown" for frame in frames]
    return "\n".join(reversed(frame_strs))


class _SourceInfoContext(threading.local):
    context: SourceInfo

    def __init__(self):
        super().__init__()
        self.context = new_source_info()


_source_info_context = _SourceInfoContext()


def current() -> SourceInfo:
    source_info = _source_info_context.context
    if not source_info.traceback:
        source_info = source_info.replace(traceback=Traceback.get_traceback())
    return source_info


class JaxStackTraceBeforeTransformation(Exception):
    pass


_message = (
    "The preceding stack trace is the source of the JAX operation that, once "
    "transformed by JAX, triggered the following exception.\n"
    "\n--------------------"
)


def has_user_context(e):
    while e is not None:
        if isinstance(e, JaxStackTraceBeforeTransformation):
            return True
        e = e.__cause__
    return False


class UserContextManager:
    __slots__ = ["traceback", "name_stack", "prev"]

    def __init__(
        self, traceback: Traceback | None, *, name_stack: NameStack | None = None
    ):
        self.traceback = traceback
        self.name_stack = name_stack

    def __enter__(self):
        self.prev = _source_info_context.context
        _source_info_context.context = _source_info_context.context.replace(
            traceback=self.traceback, name_stack=self.name_stack
        )

    def __exit__(self, exc_type, exc_value, traceback):
        _source_info_context.context = self.prev
        if exc_type is None or exc_value is None:
            return

        if self.traceback is None or has_user_context(exc_value):
            return

        filtered_tb = traceback_util.filter_traceback(
            self.traceback.as_python_traceback()
        )
        if filtered_tb:
            msg = traceback_util.format_exception_only(exc_value)
            msg = f"{msg}\n\n{_message}"
            exp = JaxStackTraceBeforeTransformation(msg).with_traceback(filtered_tb)
            exp.__context__ = exc_value.__context__
            exp.__cause__ = exc_value.__cause__
            exp.__suppress_context__ = exc_value.__suppress_context__
            exc_value.__context__ = None
            exc_value.__cause__ = exp


user_context = UserContextManager


def current_name_stack() -> NameStack:
    return _source_info_context.context.name_stack


class ExtendNameStackContextManager(contextlib.ContextDecorator):
    __slots__ = ["name", "prev"]

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.prev = prev = _source_info_context.context
        name_stack = prev.name_stack.extend(self.name)
        _source_info_context.context = prev.replace(name_stack=name_stack)
        return name_stack

    def __exit__(self, exc_type, exc_value, traceback):
        _source_info_context.context = self.prev


extend_name_stack = ExtendNameStackContextManager


class SetNameStackContextManager(contextlib.ContextDecorator):
    __slots__ = ["name_stack", "prev"]

    def __init__(self, name_stack: NameStack):
        self.name_stack = name_stack

    def __enter__(self):
        self.prev = prev = _source_info_context.context
        _source_info_context.context = prev.replace(name_stack=self.name_stack)

    def __exit__(self, exc_type, exc_value, traceback):
        _source_info_context.context = self.prev


set_name_stack = SetNameStackContextManager


# TODO(mattjj,phawkins): figure out why the commented-out reset_name_stack
# implementation doesn't work. Luckily this context manager isn't called much so
# the performance shouldn't matter. See blame commit message for repro.
# reset_name_stack = lambda: SetNameStackContextManager(NameStack())
@contextlib.contextmanager
def reset_name_stack() -> Iterator[None]:
    with set_name_stack(NameStack()):
        yield


class TransformNameStackContextManager(contextlib.ContextDecorator):
    __slots__ = ["name", "prev"]

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.prev = prev = _source_info_context.context
        name_stack = prev.name_stack.transform(self.name)
        _source_info_context.context = prev.replace(name_stack=name_stack)
        return name_stack

    def __exit__(self, exc_type, exc_value, traceback):
        _source_info_context.context = self.prev


transform_name_stack = TransformNameStackContextManager


def get_canonical_source_file(file_name: str, caches: TracebackCaches) -> str:
    canonical_file_name = caches.canonical_name_cache.get(file_name, None)
    if canonical_file_name is not None:
        return canonical_file_name

    # pattern = config.hlo_source_file_canonicalization_regex.value
    # if pattern:
    #     file_name = re.sub(pattern, "", file_name)
    caches.canonical_name_cache[file_name] = file_name
    return file_name


def _is_user_file(file_name: str) -> bool:
    is_user = _traceback_caches.is_user_file_cache.get(file_name, None)
    if is_user is not None:
        return is_user
    out = is_user_filename(file_name)
    _traceback_caches.is_user_file_cache[file_name] = out
    return out


def _traceback_to_location(tb: Traceback) -> Location:
    """Converts a full traceback to a callsite() MLIR location."""
    loc = _traceback_caches.traceback_cache.get(tb, None)
    if loc is not None:
        return loc

    frame_locs = []
    frames_limit = _traceback_in_locations_limit
    frames_limit = frames_limit if frames_limit >= 0 else 1000

    codes, lastis = tb.raw_frames()
    for i, code in enumerate(codes):
        if not _is_user_file(code.co_filename):
            continue

        lasti = lastis[i]
        code_lasti = code, lasti
        loc = _traceback_caches.location_cache.get(code_lasti, None)
        if loc is None:
            frame = raw_frame_to_frame(code, lasti)
            if (
                frame.start_column is not None
                and frame.end_line is not None
                and frame.end_column is not None
            ):
                file_loc = Location.file(
                    get_canonical_source_file(frame.file_name, _traceback_caches),
                    frame.start_line,
                    frame.start_column,
                    frame.end_line,
                    frame.end_column,
                )
            else:
                file_loc = Location.file(
                    get_canonical_source_file(frame.file_name, _traceback_caches),
                    frame.start_line,
                    frame.start_column,
                )
            loc = Location.name(frame.function_name, childLoc=file_loc)
            _traceback_caches.location_cache[code_lasti] = loc
        frame_locs.append(loc)
        if len(frame_locs) >= frames_limit:
            break

    n = len(frame_locs)
    if n == 0:
        loc = Location.unknown()
    elif n == 1:
        loc = frame_locs[0]
    else:
        loc = Location.callsite(frame_locs[0], frame_locs[1:])
    _traceback_caches.traceback_cache[tb] = loc
    return loc


def source_info_to_location(
    primitive: None,
    name_stack: NameStack,
    traceback: Traceback | None,
) -> Location:
    if _include_full_tracebacks_in_locations:
        if traceback is None:
            loc = Location.unknown()
        else:
            loc = _traceback_to_location(traceback)
    else:
        frame = user_frame(traceback)
        if frame is None:
            loc = Location.unknown()
        else:
            loc = Location.file(
                get_canonical_source_file(frame.file_name, _traceback_caches),
                frame.start_line,
                frame.start_column,
            )
    if primitive is None:
        if name_stack.stack:
            loc = Location.name(str(name_stack), childLoc=loc)
    else:
        eqn_str = (
            f"{name_stack}/{primitive.name}" if name_stack.stack else primitive.name
        )
        loc = Location.name(eqn_str, childLoc=loc)
        loc = Location.name(f"{primitive.name}:", childLoc=loc)
    return loc
