from textwrap import dedent

from mlir_opt_repl import engine
from mlir_opt_repl.diff import render_side_by_side, render_unified_diff
from mlir_opt_repl.render import BOLD, CYAN, DIM, GREEN, RED, RESET


def interactive_main():
    try:
        import readline  # noqa: F401 — enables line editing in input()
    except ModuleNotFoundError:  # pragma: no cover
        pass

    engine.check_mlir_opt()

    print(f"{BOLD}mlir-opt-repl{RESET} (using {engine.MLIR_OPT})")
    print(
        "Commands: load <file>, run <passes...>, rewind [N], history, diff, sbs, ir, reset, quit"
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
            # Validate the IR by running mlir-opt with no passes
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
            steps = int(parts[1]) if len(parts) > 1 else 1
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
            for i, (desc, _) in enumerate(engine.ir_history):
                marker = (
                    f" {GREEN}<-- current{RESET}"
                    if i == len(engine.ir_history) - 1
                    else ""
                )
                print(f"  {BOLD}[{i}]{RESET} {desc}{marker}")

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
            print(f"{GREEN}IR state cleared.{RESET}")

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
            print(
                dedent(
                    f"""\
                {BOLD}Commands:{RESET}
                  {CYAN}load <file.mlir>{RESET}    Load MLIR from a file
                  {CYAN}load -{RESET}              Load MLIR from stdin (blank line to finish)
                  {CYAN}run <passes...>{RESET}     Apply passes to current IR
                  {CYAN}ir{RESET}                  Show current IR
                  {CYAN}history{RESET}             Show pass application history
                  {CYAN}diff [a b]{RESET}          Unified diff (last step, or between indices a and b)
                  {CYAN}sbs [a b]{RESET}           Side-by-side diff (last step, or between indices a and b)
                  {CYAN}rewind [N]{RESET}          Undo last N steps (default 1)
                  {CYAN}reset{RESET}               Clear all state
                  {CYAN}passes [filter]{RESET}     List available passes
                  {CYAN}quit{RESET}                Exit
            """
                )
            )

        else:
            print(f"{RED}Unknown command: {cmd}. Type 'help' for usage.{RESET}")
