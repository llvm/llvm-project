import difflib
from dataclasses import dataclass


@dataclass(slots=True)
class EditOp:
    tag: str
    src_start: int
    src_end: int
    dst_start: int
    dst_end: int
    new_lines: tuple = ()
    old_lines: tuple = ()


@dataclass(slots=True)
class HistoryEntry:
    description: str
    full_ir: str = None
    patch: list = None
    parent_index: int = None


class IRHistory:
    def __init__(self):
        self._entries = []
        self._current_ir = None
        self._redo_stack = []

    @property
    def current_ir(self):
        return self._current_ir

    def __len__(self):
        return len(self._entries)

    def __bool__(self):
        return len(self._entries) > 0

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self._entries) + idx
        return (self._entries[idx].description, self.reconstruct_ir(idx))

    def __iter__(self):
        for i, desc, ir in self.iter_with_ir():
            yield (desc, ir)

    def get_description(self, index):
        if index < 0:
            index = len(self._entries) + index
        if index < len(self._entries):
            return self._entries[index].description
        return "?"

    def append(self, description, ir_text):
        if self._redo_stack:
            self._redo_stack = []
        if not self._entries:
            self._entries.append(
                HistoryEntry(
                    description=description, full_ir=ir_text, parent_index=None
                )
            )
        else:
            patch = self._compute_patch(self._current_ir, ir_text)
            parent_index = len(self._entries) - 1
            self._entries.append(
                HistoryEntry(
                    description=description, patch=patch, parent_index=parent_index
                )
            )
        self._current_ir = ir_text

    def clear(self):
        self._entries = []
        self._current_ir = None
        self._redo_stack = []

    def truncate(self, new_length):
        if new_length >= len(self._entries):
            return
        removed = self._entries[new_length:]
        self._redo_stack = removed
        for entry in reversed(removed):
            if entry.patch is not None:
                lines = self._current_ir.splitlines(keepends=True)
                lines = self._apply_patch_reverse(lines, entry.patch)
                self._current_ir = "".join(lines)
            else:
                self._current_ir = entry.full_ir
        self._entries = self._entries[:new_length]
        if not self._entries:
            self._current_ir = None

    def reconstruct_ir(self, index):
        if index < 0:
            index = len(self._entries) + index
        if index == len(self._entries) - 1:
            return self._current_ir
        entry = self._entries[index]
        if entry.full_ir is not None:
            return entry.full_ir
        chain = []
        i = index
        while i >= 0:
            e = self._entries[i]
            if e.full_ir is not None:
                break
            chain.append(e.patch)
            i = e.parent_index if e.parent_index is not None else i - 1
        lines = self._entries[i].full_ir.splitlines(keepends=True)
        for patch in reversed(chain):
            lines = self._apply_patch_forward(lines, patch)
        return "".join(lines)

    def iter_with_ir(self):
        if not self._entries:
            return
        current_lines = self._entries[0].full_ir.splitlines(keepends=True)
        yield (0, self._entries[0].description, self._entries[0].full_ir)
        for i in range(1, len(self._entries)):
            entry = self._entries[i]
            if entry.full_ir is not None:
                current_lines = entry.full_ir.splitlines(keepends=True)
                yield (i, entry.description, entry.full_ir)
            else:
                current_lines = self._apply_patch_forward(current_lines, entry.patch)
                yield (i, entry.description, "".join(current_lines))

    def iter_descriptions(self):
        for i, entry in enumerate(self._entries):
            yield (i, entry.description)

    def _compute_patch(self, old_text, new_text):
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        sm = difflib.SequenceMatcher(None, old_lines, new_lines)
        ops = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                ops.append(
                    EditOp(tag=tag, src_start=i1, src_end=i2, dst_start=j1, dst_end=j2)
                )
            else:
                ops.append(
                    EditOp(
                        tag=tag,
                        src_start=i1,
                        src_end=i2,
                        dst_start=j1,
                        dst_end=j2,
                        new_lines=tuple(new_lines[j1:j2]),
                        old_lines=tuple(old_lines[i1:i2]),
                    )
                )
        return ops

    def _apply_patch_forward(self, source_lines, patch):
        result = []
        for op in patch:
            if op.tag == "equal":
                result.extend(source_lines[op.src_start : op.src_end])
            elif op.tag == "replace":
                result.extend(op.new_lines)
            elif op.tag == "delete":
                pass
            elif op.tag == "insert":
                result.extend(op.new_lines)
        return result

    def _apply_patch_reverse(self, source_lines, patch):
        result = []
        for op in patch:
            if op.tag == "equal":
                result.extend(source_lines[op.dst_start : op.dst_end])
            elif op.tag == "replace":
                result.extend(op.old_lines)
            elif op.tag == "delete":
                result.extend(op.old_lines)
            elif op.tag == "insert":
                pass
        return result
