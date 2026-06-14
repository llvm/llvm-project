import readline
from textwrap import dedent

from mlir_opt_repl import engine
from mlir_opt_repl.diff import render_side_by_side, render_unified_diff
from mlir_opt_repl.render import BOLD, CYAN, DIM, GREEN, RED, RESET

COMMANDS = [
    "load",
    "run",
    "ir",
    "history",
    "diff",
    "sbs",
    "rewind",
    "reset",
    "passes",
    "save",
    "bookmark",
    "verify",
    "help",
    "quit",
    "exit",
]

_pass_names_cache = None
bookmarks = {}


def _get_pass_names():
    global _pass_names_cache
    if _pass_names_cache is None:
        _pass_names_cache = [p["name"] for p in engine.list_passes()]
    return _pass_names_cache


def _completer(text, state):
    line = readline.get_line_buffer()
    parts = line.lstrip().split()

    if len(parts) == 0 or (len(parts) == 1 and not line.endswith(" ")):
        matches = [c + " " for c in COMMANDS if c.startswith(text)]
    elif parts[0] == "run":
        pass_names = _get_pass_names()
        matches = [p + " " for p in pass_names if p.startswith(text)]
    elif parts[0] == "rewind" and bookmarks:
        matches = [b + " " for b in bookmarks if b.startswith(text)]
    else:
        matches = []

    return matches[state] if state < len(matches) else None


def interactive_main():
    engine.check_mlir_opt()

    readline.set_completer(_completer)
    readline.set_completer_delims(" ")
    readline.parse_and_bind("tab: complete")

    print(f"{BOLD}mlir-opt-repl{RESET} (using {engine.MLIR_OPT})")
    print(
        "Commands: load, run, ir, history, diff, sbs, rewind, reset, "
        "save, bookmark, verify, passes, help, quit"
    )
    print()

    while True:
        try:
            line = input(f"{CYAN}mlir-opt-repl>{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd == "load":
            if len(parts) < 2:
                print(
                    f"{RED}Usage: load <file.mlir> OR load - (read from stdin until blank line){RESET}"
                )
                continue
            if parts[1] == "-":
                print(f"{DIM}Enter MLIR (blank line to finish):{RESET}")
                mlir_lines = []
                while True:
                    try:
                        l = input()
                    except EOFError:
                        break
                    if l == "":
                        break
                    mlir_lines.append(l)
                mlir_text = "\n".join(mlir_lines)
            else:
                path = " ".join(parts[1:])
                try:
                    with open(path) as f:
                        mlir_text = f.read()
                except FileNotFoundError:
                    print(f"{RED}File not found: {path}{RESET}")
                    continue
            _, err = engine.run_mlir_opt(mlir_text, [])
            if err:
                print(f"{RED}Invalid MLIR:{RESET}")
                print(f"{RED}{err}{RESET}")
                continue
            engine.current_ir = mlir_text
            engine.ir_history = [("initial", mlir_text)]
            print(
                f"{GREEN}Loaded ({len(mlir_text)} bytes). Use 'run <passes>' to apply passes.{RESET}"
            )

        elif cmd == "run":
            if not parts[1:]:
                print(f"{RED}Usage: run <pass1> [pass2] ...{RESET}")
                continue
            if engine.current_ir is None:
                print(f"{RED}No IR loaded. Use 'load <file>' first.{RESET}")
                continue
            pipeline_str = " ".join(parts[1:])
            if "(" in pipeline_str:
                passes = [f"--pass-pipeline={pipeline_str}"]
            else:
                passes = ["--" + p.lstrip("-") for p in parts[1:]]
            output, err = engine.run_mlir_opt(engine.current_ir, passes)
            if err:
                print(f"{RED}{err}{RESET}")
                print(f"{DIM}Current IR unchanged.{RESET}")
            else:
                engine.current_ir = output
                engine.ir_history.append((" ".join(passes), output))
                print(output)

        elif cmd == "rewind":
            if not engine.ir_history:
                print(f"{RED}No history to rewind.{RESET}")
                continue
            target = parts[1] if len(parts) > 1 else "1"
            if target in bookmarks:
                idx = bookmarks[target]
                if idx >= len(engine.ir_history):
                    print(f"{RED}Bookmark '{target}' points to invalid index.{RESET}")
                    continue
                engine.ir_history = engine.ir_history[: idx + 1]
                desc, ir = engine.ir_history[-1]
                engine.current_ir = ir
                print(f"{GREEN}Rewound to bookmark '{target}' ({desc}).{RESET}")
            else:
                steps = int(target)
                if steps >= len(engine.ir_history):
                    desc, ir = engine.ir_history[0]
                    engine.ir_history = [engine.ir_history[0]]
                    engine.current_ir = ir
                    print(f"{GREEN}Rewound to beginning ({desc}).{RESET}")
                else:
                    engine.ir_history = engine.ir_history[:-steps]
                    desc, ir = engine.ir_history[-1]
                    engine.current_ir = ir
                    print(f"{GREEN}Rewound {steps} step(s). Now at: {desc}{RESET}")
            print()
            print(engine.current_ir)

        elif cmd == "history":
            if not engine.ir_history:
                print(f"{DIM}(no history){RESET}")
                continue
            bookmark_at = {v: k for k, v in bookmarks.items()}
            for i, (desc, _) in enumerate(engine.ir_history):
                marker = (
                    f" {GREEN}<-- current{RESET}"
                    if i == len(engine.ir_history) - 1
                    else ""
                )
                bm = f" {CYAN}[{bookmark_at[i]}]{RESET}" if i in bookmark_at else ""
                print(f"  {BOLD}[{i}]{RESET} {desc}{bm}{marker}")

        elif cmd == "diff":
            if len(engine.ir_history) < 2:
                print(f"{DIM}(need at least 2 history entries){RESET}")
                continue
            if len(parts) == 3:
                a, b = int(parts[1]), int(parts[2])
            else:
                a, b = len(engine.ir_history) - 2, len(engine.ir_history) - 1
            if a < 0 or b >= len(engine.ir_history):
                print(f"{RED}Invalid indices (0..{len(engine.ir_history)-1}){RESET}")
                continue
            print(
                render_unified_diff(
                    engine.ir_history[a][1].splitlines(),
                    engine.ir_history[b][1].splitlines(),
                    engine.ir_history[a][0],
                    engine.ir_history[b][0],
                    pretty=True,
                )
            )

        elif cmd == "sbs":
            if len(engine.ir_history) < 2:
                print(f"{DIM}(need at least 2 history entries){RESET}")
                continue
            if len(parts) == 3:
                a, b = int(parts[1]), int(parts[2])
            else:
                a, b = len(engine.ir_history) - 2, len(engine.ir_history) - 1
            if a < 0 or b >= len(engine.ir_history):
                print(f"{RED}Invalid indices (0..{len(engine.ir_history)-1}){RESET}")
                continue
            print(
                render_side_by_side(
                    engine.ir_history[a][1].splitlines(),
                    engine.ir_history[b][1].splitlines(),
                    engine.ir_history[a][0],
                    engine.ir_history[b][0],
                    pretty=True,
                )
            )

        elif cmd == "ir":
            if engine.current_ir is None:
                print(f"{DIM}(no IR state set){RESET}")
            else:
                print(engine.current_ir)

        elif cmd == "reset":
            engine.current_ir = None
            engine.ir_history = []
            bookmarks.clear()
            print(f"{GREEN}IR state cleared.{RESET}")

        elif cmd == "save":
            if len(parts) < 2:
                print(f"{RED}Usage: save <file.mlir>{RESET}")
                continue
            if engine.current_ir is None:
                print(f"{RED}No IR to save.{RESET}")
                continue
            path = " ".join(parts[1:])
            with open(path, "w") as f:
                f.write(engine.current_ir)
                f.write("\n")
            print(f"{GREEN}Saved to {path}{RESET}")

        elif cmd == "bookmark":
            if len(parts) < 2:
                if not bookmarks:
                    print(f"{DIM}(no bookmarks){RESET}")
                else:
                    for name, idx in sorted(bookmarks.items(), key=lambda x: x[1]):
                        desc = (
                            engine.ir_history[idx][0]
                            if idx < len(engine.ir_history)
                            else "?"
                        )
                        print(f"  {BOLD}{name}{RESET} -> [{idx}] {desc}")
                continue
            name = parts[1]
            idx = len(engine.ir_history) - 1
            bookmarks[name] = idx
            print(f"{GREEN}Bookmarked [{idx}] as '{name}'{RESET}")

        elif cmd == "verify":
            if engine.current_ir is None:
                print(f"{RED}No IR loaded.{RESET}")
                continue
            _, err = engine.run_mlir_opt(engine.current_ir, ["--verify-diagnostics"])
            if err:
                print(f"{RED}{err}{RESET}")
            else:
                print(f"{GREEN}IR is valid.{RESET}")

        elif cmd == "passes":
            filt = parts[1] if len(parts) > 1 else ""
            all_passes = engine.list_passes()
            if filt:
                all_passes = [
                    p
                    for p in all_passes
                    if filt.lower() in p.get("name", "").lower()
                    or filt.lower() in p.get("description", "").lower()
                ]
            for p in all_passes:
                print(f"  {BOLD}--{p['name']}{RESET}: {DIM}{p['description']}{RESET}")
            if not all_passes:
                print(f"{DIM}(no passes matched){RESET}")

        elif cmd == "help":
            print(dedent(f"""\
                    {BOLD}Commands:{RESET}
                      {CYAN}load <file.mlir>{RESET}    Load MLIR from a file
                      {CYAN}load -{RESET}              Load MLIR from stdin (blank line to finish)
                      {CYAN}run <passes...>{RESET}     Apply passes to current IR
                      {CYAN}run <pipeline>{RESET}      Apply a pass-pipeline string (e.g. builtin.module(...))
                      {CYAN}ir{RESET}                  Show current IR
                      {CYAN}history{RESET}             Show pass application history
                      {CYAN}diff [a b]{RESET}          Unified diff (last step, or between indices a and b)
                      {CYAN}sbs [a b]{RESET}           Side-by-side diff (last step, or between indices a and b)
                      {CYAN}rewind [N|name]{RESET}     Undo last N steps or rewind to a bookmark
                      {CYAN}bookmark [name]{RESET}     Bookmark current step (no arg: list bookmarks)
                      {CYAN}save <file>{RESET}         Save current IR to a file
                      {CYAN}verify{RESET}              Verify current IR is valid
                      {CYAN}reset{RESET}               Clear all state
                      {CYAN}passes [filter]{RESET}     List available passes (tab-completable)
                      {CYAN}quit{RESET}                Exit
                """))

        else:
            print(f"{RED}Unknown command: {cmd}. Type 'help' for usage.{RESET}")
