import os
import shutil
import subprocess
import sys
import tempfile

from mlir_opt_repl.diff import render_side_by_side, render_unified_diff
from mlir_opt_repl.history import IRHistory

MLIR_OPT = os.environ.get("MLIR_OPT", "mlir-opt")


def check_mlir_opt():
    if not shutil.which(MLIR_OPT):
        print(
            f"error: mlir-opt not found at '{MLIR_OPT}'. "
            "Set MLIR_OPT to the path of your mlir-opt binary.",
            file=sys.stderr,
        )
        sys.exit(1)


class Engine:
    def __init__(self):
        self._ir_history = IRHistory()
        self.bookmarks = {}

    def get_current_ir(self):
        return self._ir_history.current_ir

    def history_len(self):
        return len(self._ir_history)

    def history_empty(self):
        return not self._ir_history

    def history_get(self, index):
        return self._ir_history[index]

    def history_get_description(self, index):
        return self._ir_history.get_description(index)

    def history_append(self, desc, ir_text):
        self._ir_history.append(desc, ir_text)

    def history_truncate(self, n):
        self._ir_history.truncate(n)

    def history_clear(self):
        self._ir_history.clear()

    def history_iter_with_ir(self):
        return self._ir_history.iter_with_ir()

    def history_iter_descriptions(self):
        return self._ir_history.iter_descriptions()


state = Engine()


def _build_pass_args(passes, extra_args=None):
    if len(passes) == 1 and "(" in passes[0]:
        args = [f"--pass-pipeline={passes[0]}"]
    else:
        args = ["--" + p.lstrip("-") for p in passes]
    if extra_args:
        args += ["--" + a.lstrip("-") for a in extra_args]
    return args


def run_mlir_opt(ir_text, args):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(ir_text)
        f.flush()
        tmp_path = f.name

    try:
        cmd = [MLIR_OPT] + args + [tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None, result.stderr.strip()
        return result.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return None, "mlir-opt timed out after 30 seconds"
    except FileNotFoundError:
        return None, f"mlir-opt not found at: {MLIR_OPT}"
    finally:
        os.unlink(tmp_path)


def get_help_text():
    try:
        result = subprocess.run(
            [MLIR_OPT, "--help-hidden"], capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except Exception as e:
        return f"error: {e}"


def list_passes():
    try:
        result = subprocess.run(
            [MLIR_OPT, "--help"], capture_output=True, text=True, timeout=10
        )
        passes = []
        for line in result.stdout.splitlines():
            if not line.startswith("      --"):
                continue
            stripped = line.strip()
            parts = stripped.split(" - ", 1)
            name = parts[0].strip().lstrip("-")
            desc = parts[1].strip() if len(parts) > 1 else ""
            passes.append({"name": name, "description": desc})
        return passes
    except Exception as e:
        return [{"error": str(e)}]


def handle_tool_call(name, arguments):
    if name == "run_pipeline":
        mlir = arguments["mlir"]
        passes = _build_pass_args(arguments["passes"], arguments.get("extra_args"))
        output, err = run_mlir_opt(mlir, passes)
        if err:
            return {
                "content": [{"type": "text", "text": f"Error:\n{err}"}],
                "isError": True,
            }
        state.history_clear()
        state.history_append("initial", mlir)
        state.history_append(" ".join(passes), output)
        return {"content": [{"type": "text", "text": output}]}

    elif name == "chain_pipeline":
        if state.get_current_ir() is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: no current IR state. Use run_pipeline first.",
                    }
                ],
                "isError": True,
            }
        passes = _build_pass_args(arguments["passes"], arguments.get("extra_args"))
        output, err = run_mlir_opt(state.get_current_ir(), passes)
        if err:
            return {
                "content": [
                    {"type": "text", "text": f"Error:\n{err}\n\nCurrent IR unchanged."}
                ],
                "isError": True,
            }
        state.history_append(" ".join(passes), output)
        return {"content": [{"type": "text", "text": output}]}

    elif name == "get_current_ir":
        if state.get_current_ir() is None:
            return {"content": [{"type": "text", "text": "(no IR state set)"}]}
        return {"content": [{"type": "text", "text": state.get_current_ir()}]}

    elif name == "reset":
        state.history_clear()
        state.bookmarks = {}
        return {"content": [{"type": "text", "text": "IR state cleared."}]}

    elif name == "rewind":
        if state.history_empty():
            return {
                "content": [{"type": "text", "text": "Error: no history to rewind."}],
                "isError": True,
            }
        target = arguments.get("target")
        steps = arguments.get("steps", 1)
        if target and target in state.bookmarks:
            idx = state.bookmarks[target]
            if idx >= state.history_len():
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: bookmark '{target}' points to invalid index.",
                        }
                    ],
                    "isError": True,
                }
            state.history_truncate(idx + 1)
            desc = state.history_get_description(-1)
            ir = state.get_current_ir()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Rewound to bookmark '{target}' ({desc}).\n\n{ir}",
                    }
                ]
            }
        if steps >= state.history_len():
            state.history_truncate(1)
            desc = state.history_get_description(0)
            ir = state.get_current_ir()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Rewound to beginning ({desc}).\n\n{ir}",
                    }
                ]
            }
        state.history_truncate(state.history_len() - steps)
        desc = state.history_get_description(-1)
        ir = state.get_current_ir()
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Rewound {steps} step(s). Now at: {desc}\n\n{ir}",
                }
            ]
        }

    elif name == "bookmark":
        if state.history_empty():
            return {
                "content": [{"type": "text", "text": "Error: no history to bookmark."}],
                "isError": True,
            }
        bm_name = arguments.get("name", "")
        if not bm_name:
            if not state.bookmarks:
                return {"content": [{"type": "text", "text": "(no bookmarks)"}]}
            lines = []
            for n, idx in sorted(state.bookmarks.items(), key=lambda x: x[1]):
                desc = state.history_get_description(idx)
                lines.append(f"  {n} -> [{idx}] {desc}")
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}
        idx = state.history_len() - 1
        state.bookmarks[bm_name] = idx
        return {
            "content": [{"type": "text", "text": f"Bookmarked [{idx}] as '{bm_name}'"}]
        }

    elif name == "save":
        if state.get_current_ir() is None:
            return {
                "content": [{"type": "text", "text": "Error: no IR to save."}],
                "isError": True,
            }
        path = arguments.get("path", "")
        if not path:
            return {
                "content": [{"type": "text", "text": "Error: path is required."}],
                "isError": True,
            }
        with open(path, "w") as f:
            f.write(state.get_current_ir())
            f.write("\n")
        return {"content": [{"type": "text", "text": f"Saved to {path}"}]}

    elif name == "verify":
        if state.get_current_ir() is None:
            return {
                "content": [{"type": "text", "text": "Error: no IR to verify."}],
                "isError": True,
            }
        _, err = run_mlir_opt(state.get_current_ir(), ["--verify-diagnostics"])
        if err:
            return {
                "content": [{"type": "text", "text": f"Verification failed:\n{err}"}],
                "isError": True,
            }
        return {"content": [{"type": "text", "text": "IR is valid."}]}

    elif name == "history":
        if state.history_empty():
            return {"content": [{"type": "text", "text": "(no history)"}]}
        show_ir = arguments.get("show_ir", False)
        fmt = arguments.get("format")
        pretty = arguments.get("pretty", False)
        width = arguments.get("width")
        lines = []
        bookmark_at = {v: k for k, v in state.bookmarks.items()}
        prev_ir = None
        for i, desc, ir in state.history_iter_with_ir():
            marker = " <-- current" if i == state.history_len() - 1 else ""
            bm = f" [{bookmark_at[i]}]" if i in bookmark_at else ""
            lines.append(f"[{i}] {desc}{bm}{marker}")
            if fmt == "side_by_side" and i > 0 and prev_ir is not None:
                sbs = render_side_by_side(
                    prev_ir.splitlines(),
                    ir.splitlines(),
                    state.history_get_description(i - 1),
                    desc,
                    width=width,
                    pretty=pretty,
                )
                lines.append(sbs)
                lines.append("")
            elif fmt == "unified" and i > 0 and prev_ir is not None:
                lines.append(
                    render_unified_diff(
                        prev_ir.splitlines(),
                        ir.splitlines(),
                        state.history_get_description(i - 1),
                        desc,
                        pretty=pretty,
                    )
                )
                lines.append("")
            elif show_ir:
                lines.append(ir)
                lines.append("")
            prev_ir = ir
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    elif name == "list_passes":
        text = get_help_text()
        return {"content": [{"type": "text", "text": text}]}

    return {
        "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
        "isError": True,
    }
