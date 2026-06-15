import readline
from textwrap import dedent

from mlir_opt_repl.engine import (
    MLIR_OPT,
    check_mlir_opt,
    state,
    list_passes,
    run_mlir_opt,
)
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
        _pass_names_cache = [p["name"] for p in list_passes()]
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
    check_mlir_opt()

    readline.set_completer(_completer)
    readline.set_completer_delims(" ")
    readline.parse_and_bind("tab: complete")

    print(f"{BOLD}mlir-opt-repl{RESET} (using {MLIR_OPT})")
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
            _, err = run_mlir_opt(mlir_text, [])
            if err:
                print(f"{RED}Invalid MLIR:{RESET}")
                print(f"{RED}{err}{RESET}")
                continue
            state.history_clear()
            state.history_append("initial", mlir_text)
            print(
                f"{GREEN}Loaded ({len(mlir_text)} bytes). Use 'run <passes>' to apply passes.{RESET}"
            )

        elif cmd == "run":
            if not parts[1:]:
                print(f"{RED}Usage: run <pass1> [pass2] ...{RESET}")
                continue
            if state.get_current_ir() is None:
                print(f"{RED}No IR loaded. Use 'load <file>' first.{RESET}")
                continue
            pipeline_str = " ".join(parts[1:])
            if "(" in pipeline_str:
                passes = [f"--pass-pipeline={pipeline_str}"]
            else:
                passes = ["--" + p.lstrip("-") for p in parts[1:]]
            output, err = run_mlir_opt(state.get_current_ir(), passes)
            if err:
                print(f"{RED}{err}{RESET}")
                print(f"{DIM}Current IR unchanged.{RESET}")
            else:
                state.history_append(" ".join(passes), output)
                print(output)

        elif cmd == "rewind":
            if state.history_empty():
                print(f"{RED}No history to rewind.{RESET}")
                continue
            target = parts[1] if len(parts) > 1 else "1"
            if target in bookmarks:
                idx = bookmarks[target]
                if idx >= state.history_len():
                    print(f"{RED}Bookmark '{target}' points to invalid index.{RESET}")
                    continue
                state.history_truncate(idx + 1)
                desc = state.history_get_description(-1)
                print(f"{GREEN}Rewound to bookmark '{target}' ({desc}).{RESET}")
            else:
                steps = int(target)
                if steps >= state.history_len():
                    state.history_truncate(1)
                    desc = state.history_get_description(0)
                    print(f"{GREEN}Rewound to beginning ({desc}).{RESET}")
                else:
                    state.history_truncate(state.history_len() - steps)
                    desc = state.history_get_description(-1)
                    print(f"{GREEN}Rewound {steps} step(s). Now at: {desc}{RESET}")
            print()
            print(state.get_current_ir())

        elif cmd == "history":
            if state.history_empty():
                print(f"{DIM}(no history){RESET}")
                continue
            bookmark_at = {v: k for k, v in bookmarks.items()}
            for i, desc in state.history_iter_descriptions():
                marker = (
                    f" {GREEN}<-- current{RESET}"
                    if i == state.history_len() - 1
                    else ""
                )
                bm = f" {CYAN}[{bookmark_at[i]}]{RESET}" if i in bookmark_at else ""
                print(f"  {BOLD}[{i}]{RESET} {desc}{bm}{marker}")

        elif cmd == "diff":
            if state.history_len() < 2:
                print(f"{DIM}(need at least 2 history entries){RESET}")
                continue
            if len(parts) == 3:
                a, b = int(parts[1]), int(parts[2])
            else:
                a, b = state.history_len() - 2, state.history_len() - 1
            if a < 0 or b >= state.history_len():
                print(f"{RED}Invalid indices (0..{state.history_len()-1}){RESET}")
                continue
            print(
                render_unified_diff(
                    state.history_get(a)[1].splitlines(),
                    state.history_get(b)[1].splitlines(),
                    state.history_get_description(a),
                    state.history_get_description(b),
                    pretty=True,
                )
            )

        elif cmd == "sbs":
            if state.history_len() < 2:
                print(f"{DIM}(need at least 2 history entries){RESET}")
                continue
            if len(parts) == 3:
                a, b = int(parts[1]), int(parts[2])
            else:
                a, b = state.history_len() - 2, state.history_len() - 1
            if a < 0 or b >= state.history_len():
                print(f"{RED}Invalid indices (0..{state.history_len()-1}){RESET}")
                continue
            print(
                render_side_by_side(
                    state.history_get(a)[1].splitlines(),
                    state.history_get(b)[1].splitlines(),
                    state.history_get_description(a),
                    state.history_get_description(b),
                    pretty=True,
                )
            )

        elif cmd == "ir":
            if state.get_current_ir() is None:
                print(f"{DIM}(no IR state set){RESET}")
            else:
                print(state.get_current_ir())

        elif cmd == "reset":
            state.history_clear()
            state.bookmarks = {}
            bookmarks.clear()
            print(f"{GREEN}IR state cleared.{RESET}")

        elif cmd == "save":
            if len(parts) < 2:
                print(f"{RED}Usage: save <file.mlir>{RESET}")
                continue
            if state.get_current_ir() is None:
                print(f"{RED}No IR to save.{RESET}")
                continue
            path = " ".join(parts[1:])
            with open(path, "w") as f:
                f.write(state.get_current_ir())
                f.write("\n")
            print(f"{GREEN}Saved to {path}{RESET}")

        elif cmd == "bookmark":
            if len(parts) < 2:
                if not bookmarks:
                    print(f"{DIM}(no bookmarks){RESET}")
                else:
                    for name, idx in sorted(bookmarks.items(), key=lambda x: x[1]):
                        desc = state.history_get_description(idx)
                        print(f"  {BOLD}{name}{RESET} -> [{idx}] {desc}")
                continue
            name = parts[1]
            idx = state.history_len() - 1
            bookmarks[name] = idx
            print(f"{GREEN}Bookmarked [{idx}] as '{name}'{RESET}")

        elif cmd == "verify":
            if state.get_current_ir() is None:
                print(f"{RED}No IR loaded.{RESET}")
                continue
            _, err = run_mlir_opt(state.get_current_ir(), ["--verify-diagnostics"])
            if err:
                print(f"{RED}{err}{RESET}")
            else:
                print(f"{GREEN}IR is valid.{RESET}")

        elif cmd == "passes":
            filt = parts[1] if len(parts) > 1 else ""
            all_passes = list_passes()
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
