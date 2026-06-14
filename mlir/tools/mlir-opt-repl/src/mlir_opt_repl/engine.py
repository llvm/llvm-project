import os
import shutil
import subprocess
import sys
import tempfile

from mlir_opt_repl.diff import render_side_by_side, render_unified_diff

MLIR_OPT = os.environ.get("MLIR_OPT", "mlir-opt")


def check_mlir_opt():
    if not shutil.which(MLIR_OPT):
        print(
            f"error: mlir-opt not found at '{MLIR_OPT}'. "
            "Set MLIR_OPT to the path of your mlir-opt binary.",
            file=sys.stderr,
        )
        sys.exit(1)


current_ir = None
ir_history = []
bookmarks = {}


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


def list_passes():
    try:
        result = subprocess.run(
            [MLIR_OPT, "--help"], capture_output=True, text=True, timeout=10
        )
        passes = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if (
                stripped.startswith("--convert-")
                or stripped.startswith("--canonicalize")
                or stripped.startswith("--cse")
                or stripped.startswith("--inline")
                or stripped.startswith("--mem2reg")
                or stripped.startswith("--symbol-")
            ):
                parts = stripped.split(" - ", 1)
                name = parts[0].strip().lstrip("-")
                desc = parts[1].strip() if len(parts) > 1 else ""
                passes.append({"name": name, "description": desc})
        return passes
    except Exception as e:
        return [{"error": str(e)}]


def handle_tool_call(name, arguments):
    global current_ir, ir_history, bookmarks

    if name == "run_pipeline":
        mlir = arguments["mlir"]
        passes = _build_pass_args(arguments["passes"], arguments.get("extra_args"))
        output, err = run_mlir_opt(mlir, passes)
        if err:
            return {
                "content": [{"type": "text", "text": f"Error:\n{err}"}],
                "isError": True,
            }
        ir_history = [("initial", mlir)]
        current_ir = output
        ir_history.append((" ".join(passes), output))
        return {"content": [{"type": "text", "text": output}]}

    elif name == "chain_pipeline":
        if current_ir is None:
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
        output, err = run_mlir_opt(current_ir, passes)
        if err:
            return {
                "content": [
                    {"type": "text", "text": f"Error:\n{err}\n\nCurrent IR unchanged."}
                ],
                "isError": True,
            }
        current_ir = output
        ir_history.append((" ".join(passes), output))
        return {"content": [{"type": "text", "text": output}]}

    elif name == "get_current_ir":
        if current_ir is None:
            return {"content": [{"type": "text", "text": "(no IR state set)"}]}
        return {"content": [{"type": "text", "text": current_ir}]}

    elif name == "reset":
        current_ir = None
        ir_history = []
        bookmarks = {}
        return {"content": [{"type": "text", "text": "IR state cleared."}]}

    elif name == "rewind":
        if not ir_history:
            return {
                "content": [{"type": "text", "text": "Error: no history to rewind."}],
                "isError": True,
            }
        target = arguments.get("target")
        steps = arguments.get("steps", 1)
        if target and target in bookmarks:
            idx = bookmarks[target]
            if idx >= len(ir_history):
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: bookmark '{target}' points to invalid index.",
                        }
                    ],
                    "isError": True,
                }
            ir_history = ir_history[: idx + 1]
            desc, ir = ir_history[-1]
            current_ir = ir
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Rewound to bookmark '{target}' ({desc}).\n\n{ir}",
                    }
                ]
            }
        if steps >= len(ir_history):
            desc, ir = ir_history[0]
            ir_history = [ir_history[0]]
            current_ir = ir
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Rewound to beginning ({desc}).\n\n{ir}",
                    }
                ]
            }
        ir_history = ir_history[:-steps]
        desc, ir = ir_history[-1]
        current_ir = ir
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Rewound {steps} step(s). Now at: {desc}\n\n{ir}",
                }
            ]
        }

    elif name == "bookmark":
        if not ir_history:
            return {
                "content": [{"type": "text", "text": "Error: no history to bookmark."}],
                "isError": True,
            }
        bm_name = arguments.get("name", "")
        if not bm_name:
            if not bookmarks:
                return {"content": [{"type": "text", "text": "(no bookmarks)"}]}
            lines = []
            for n, idx in sorted(bookmarks.items(), key=lambda x: x[1]):
                desc = ir_history[idx][0] if idx < len(ir_history) else "?"
                lines.append(f"  {n} -> [{idx}] {desc}")
            return {"content": [{"type": "text", "text": "\n".join(lines)}]}
        idx = len(ir_history) - 1
        bookmarks[bm_name] = idx
        return {
            "content": [{"type": "text", "text": f"Bookmarked [{idx}] as '{bm_name}'"}]
        }

    elif name == "save":
        if current_ir is None:
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
            f.write(current_ir)
            f.write("\n")
        return {"content": [{"type": "text", "text": f"Saved to {path}"}]}

    elif name == "verify":
        if current_ir is None:
            return {
                "content": [{"type": "text", "text": "Error: no IR to verify."}],
                "isError": True,
            }
        _, err = run_mlir_opt(current_ir, ["--verify-diagnostics"])
        if err:
            return {
                "content": [{"type": "text", "text": f"Verification failed:\n{err}"}],
                "isError": True,
            }
        return {"content": [{"type": "text", "text": "IR is valid."}]}

    elif name == "history":
        if not ir_history:
            return {"content": [{"type": "text", "text": "(no history)"}]}
        show_ir = arguments.get("show_ir", False)
        fmt = arguments.get("format")
        pretty = arguments.get("pretty", False)
        width = arguments.get("width")
        lines = []
        bookmark_at = {v: k for k, v in bookmarks.items()}
        for i, (desc, ir) in enumerate(ir_history):
            marker = " <-- current" if i == len(ir_history) - 1 else ""
            bm = f" [{bookmark_at[i]}]" if i in bookmark_at else ""
            lines.append(f"[{i}] {desc}{bm}{marker}")
            if fmt == "side_by_side" and i > 0:
                prev_ir = ir_history[i - 1][1]
                sbs = render_side_by_side(
                    prev_ir.splitlines(),
                    ir.splitlines(),
                    ir_history[i - 1][0],
                    desc,
                    width=width,
                    pretty=pretty,
                )
                lines.append(sbs)
                lines.append("")
            elif fmt == "unified" and i > 0:
                prev_ir = ir_history[i - 1][1]
                lines.append(
                    render_unified_diff(
                        prev_ir.splitlines(),
                        ir.splitlines(),
                        ir_history[i - 1][0],
                        desc,
                        pretty=pretty,
                    )
                )
                lines.append("")
            elif show_ir:
                lines.append(ir)
                lines.append("")
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    elif name == "list_passes":
        filt = arguments.get("filter", "")
        passes = list_passes()
        if filt:
            passes = [
                p
                for p in passes
                if filt.lower() in p.get("name", "").lower()
                or filt.lower() in p.get("description", "").lower()
            ]
        text = "\n".join(f"--{p['name']}: {p['description']}" for p in passes)
        return {"content": [{"type": "text", "text": text or "(no passes matched)"}]}

    return {
        "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
        "isError": True,
    }
