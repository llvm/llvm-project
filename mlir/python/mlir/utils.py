from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import (
    Any,
    Callable,
    Concatenate,
    Iterator,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
)

from mlir import ir
from mlir._mlir_libs import get_dialect_registry
from mlir.dialects import func
from mlir.dialects.transform import interpreter
from mlir.passmanager import PassManager

RT = TypeVar("RT")
Param = ParamSpec("Param")


@contextmanager
def using_mlir_context(
    *,
    required_dialects: Optional[Sequence[str]] = None,
    required_extension_operations: Optional[Sequence[str]] = None,
    registration_funcs: Optional[Sequence[Callable[[ir.DialectRegistry], None]]] = None,
) -> Iterator[None]:
    """Ensure a valid context exists by creating one if necessary.

    NOTE: If values that are attached to a Context should outlive this
          contextmanager, use caller_mlir_context!

    This can be used as a function decorator or managed context in a with statement.
    The context will throw an error if the required dialects have not been registered,
    and a context is guaranteed to exist in this scope.

    This only works on dialects and not dialect extensions currently.

    Parameters
    ------------
        required_dialects:
            Dialects that need to be registered in the context
        required_extension_operations:
            Required operations by their fully specified name. These are a proxy for detecting needed dialect extensions.
        registration_funcs:
            Functions that should be called to register all missing dialects/operations if they have not been registered.
    """
    dialects = required_dialects or []
    extension_operations = required_extension_operations or []
    registrations = registration_funcs or []
    new_context = nullcontext if ir.Context.current else ir.Context
    with new_context(), ir.Location.unknown():
        context = ir.Context.current
        # Attempt to disable multithreading. This could fail if currently being
        # used in multiple threads. This must be done before checking for
        # dialects or registering dialects as both will assert fail in a
        # multithreaded situation.
        multithreading = context.is_multithreading_enabled
        if multithreading:
            context.enable_multithreading(False)

        def attempt_registration():
            """Register everything from registration_funcs."""
            nonlocal context, registrations

            # Gather dialects and extensions then add them to the context.
            registry = ir.DialectRegistry()
            for rf in registrations:
                rf(registry)

            context.append_dialect_registry(registry)

        # See if any dialects are missing, register if they are, and then assert they are all registered.
        try:
            for dialect in dialects:
                # If the dialect is registered, continue checking
                context.get_dialect_descriptor(dialect)
        except Exception:
            attempt_registration()

        for dialect in dialects:
            # If the dialect is registered, continue checking
            assert context.get_dialect_descriptor(
                dialect
            ), f"required dialect {dialect} not registered by registration_funcs"

        # See if any operations are missing and register if they are. We cannot
        # assert the operations exist in the registry after for some reason.
        #
        # TODO: Make this work for dialect extensions specifically
        for operation in extension_operations:
            # If the operation is registered, attempt to register and then strongly assert it was added
            if not context.is_registered_operation(operation):
                attempt_registration()
                break
        for operation in extension_operations:
            # First get the dialect descriptior which loads the dialect as a side effect
            dialect = operation.split(".")[0]
            assert context.get_dialect_descriptor(dialect), f"Never loaded {dialect}"
            assert context.is_registered_operation(
                operation
            ), f"expected {operation} to be registered in its dialect"
        context.enable_multithreading(multithreading)

        # Context manager related yield
        try:
            yield
        finally:
            pass


@contextmanager
def caller_mlir_context(
    *,
    required_dialects: Optional[Sequence[str]] = None,
    required_extension_operations: Optional[Sequence[str]] = None,
    registration_funcs: Optional[Sequence[Callable[[ir.DialectRegistry], None]]] = None,
) -> Iterator[None]:
    """Requires an enclosing context from the caller and ensures relevant operations are loaded.

    NOTE: If the Context is only needed inside of this contextmanager and returned values
          don't need to the Context, use using_mlir_context!

    A context must already exist before this frame is executed to ensure that any values
    continue to live on exit. Conceptually, this prevents use-after-free issues and
    makes the intention clear when one intends to return values tied to a Context.
    """
    assert (
        ir.Context.current
    ), "Caller must have a context so it outlives this function call."
    with using_mlir_context(
        required_dialects=required_dialects,
        required_extension_operations=required_extension_operations,
        registration_funcs=registration_funcs,
    ):
        # Context manager related yield
        try:
            yield
        finally:
            pass


def with_toplevel_context(f: Callable[Param, RT]) -> Callable[Param, RT]:
    """Decorate the function to be executed with a fresh MLIR context.

    This decorator will ensure the function is executed inside a context manager for a
    new MLIR context with upstream and IREE dialects registered. Note that each call to
    such a function has a new context, meaning that context-owned objects from these
    functions will not be equal to each other. All arguments and keyword arguments are
    forwarded.

    The context is destroyed before the function exits so any result from the function
    must not depend on the context.
    """

    @wraps(f)
    def decorator(*args: Param.args, **kwargs: Param.kwargs) -> RT:
        # Appending dialect registry and loading all available dialects occur on
        # context creation because of the "_site_initialize" call.
        with ir.Context(), ir.Location.unknown():
            results = f(*args, **kwargs)
        return results

    return decorator


def with_toplevel_context_create_module(
    f: Callable[Concatenate[ir.Module, Param], RT],
) -> Callable[Param, RT]:
    """Decorate function to be executed in a fresh MLIR context and give it a module.

    The decorated function will receive, as its leading argument, a fresh MLIR module.
    The context manager is set up to insert operations into this module. All other
    arguments and keyword arguments are forwarded.

    The module and context are destroyed before the function exists so any result from
    the function must not depend on either.
    """

    @with_toplevel_context
    @wraps(f)
    def internal(*args: Param.args, **kwargs: Param.kwargs) -> RT:
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            results = f(module, *args, **kwargs)
        return results

    return internal


def call_with_toplevel_context(f: Callable[[], RT]) -> Callable[[], RT]:
    """Immediately call the function in a fresh MLIR context."""
    decorated = with_toplevel_context(f)
    decorated()
    return decorated


def call_with_toplevel_context_create_module(
    f: Callable[[ir.Module], RT],
) -> Callable[[], RT]:
    """Immediately call the function in a fresh MLIR context and give it a module.

    The decorated function will receive, as its only argument, a fresh MLIR module. The
    context manager is set up to insert operations into this module.
    """
    decorated = with_toplevel_context_create_module(f)
    decorated()
    return decorated


def _debug_flags_impl(flags: Sequence[str]) -> Iterator[None]:
    from mlir.ir import _GlobalDebug

    # Save the original debug state. The debug flags will be popped rather than
    # manually copied and saved for later.
    original_flag = _GlobalDebug.flag
    _GlobalDebug.flag = True
    _GlobalDebug.push_debug_only_flags(flags)

    try:
        yield
    finally:
        # Reset the global debug flag and remove the most recent flags that were
        # appended. This assumes that nothing else popped when it should not have.
        _GlobalDebug.flag = original_flag
        _GlobalDebug.pop_debug_only_flags()


@contextmanager
def debug_flags_context(flags: Sequence[str]):
    """Temporarily create a context that enables debugging with specified filters.

    These would be the same as running with -debug-only=*flags. Where multiple contexts
    will be joined together to create the full list if they are nested.

    This requires that the core MLIR units were compiled without NDEBUG.
    """
    return _debug_flags_impl(flags)


@contextmanager
def debug_conversion(flags: Sequence[str] = []) -> Iterator[None]:
    """Temporarily create a context that enables full conversion debugging,
    potentially with additional specified filters.

    These would be the same as running with -debug-only=*flags. Where multiple contexts
    will be joined together to create the full list if they are nested.

    This requires that the core MLIR units were compiled without NDEBUG.
    """
    return _debug_flags_impl(list(flags) + ["dialect-conversion"])


@contextmanager
def debug_greedy_rewriter(flags: Sequence[str] = []) -> Iterator[None]:
    """Temporarily create a context that enables full conversion debugging,
    potentially with additional specified filters.

    These would be the same as running with -debug-only=*flags. Where multiple contexts
    will be joined together to create the full list if they are nested.

    This requires that the core MLIR units were compiled without NDEBUG.
    """
    return _debug_flags_impl(list(flags) + ["greedy_rewriter"])


@contextmanager
def debug_td(flags: Sequence[str] = [], *, full_debug: bool = False) -> Iterator[None]:
    """Temporarily create a context that enables full transform dialect debugging,
    potentially with additional specified filters.

    These would be the same as running with -debug-only=*flags. Where multiple contexts
    will be joined together to create the full list if they are nested.

    This requires that the core MLIR units were compiled without NDEBUG.
    """
    return _debug_flags_impl(
        list(flags)
        + [
            "transform-dialect",
            "transform-dialect-print-top-level-after-all",
        ]
        + (["transform-dialect-full"] if full_debug else [])
    )
