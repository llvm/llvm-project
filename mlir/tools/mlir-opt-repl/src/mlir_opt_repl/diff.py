import difflib
import shutil

from mlir_opt_repl.render import BOLD, CYAN, DIM, GREEN, RED, RESET


def render_side_by_side(
    left_lines, right_lines, left_title, right_title, width=None, pretty=True
):
    if width is None:
        width = shutil.get_terminal_size((120, 24)).columns
    col_width = (width - 3) // 2

    def truncate(s, w):
        if len(s) <= w:
            return s + " " * (w - len(s))
        return s[: w - 1] + "…"

    if pretty:
        r, g, c, d, b, rst = RED, GREEN, CYAN, DIM, BOLD, RESET
    else:
        r, g, c, d, b, rst = "", "", "", "", "", ""

    lines = []
    header = (
        f"{b}{c}{truncate(left_title, col_width)}{rst}"
        f" {d}│{rst} "
        f"{b}{c}{truncate(right_title, col_width)}{rst}"
    )
    lines.append(header)
    sep_char = "─" if pretty else "-"
    mid_char = "┼" if pretty else "+"
    div_char = "│" if pretty else "|"
    lines.append(d + sep_char * col_width + mid_char + sep_char * (col_width + 2) + rst)

    opcodes = difflib.SequenceMatcher(None, left_lines, right_lines).get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                l = truncate(left_lines[i], col_width)
                rv = truncate(right_lines[j], col_width)
                lines.append(f"{d}{l}{rst} {d}{div_char}{rst} {d}{rv}{rst}")
        elif tag == "replace":
            max_len = max(i2 - i1, j2 - j1)
            for k in range(max_len):
                if k < i2 - i1:
                    l = f"{r}{truncate(left_lines[i1 + k], col_width)}{rst}"
                else:
                    l = " " * col_width
                if k < j2 - j1:
                    rv = f"{g}{truncate(right_lines[j1 + k], col_width)}{rst}"
                else:
                    rv = " " * col_width
                lines.append(f"{l} {d}{div_char}{rst} {rv}")
        elif tag == "delete":
            for i in range(i1, i2):
                l = f"{r}{truncate(left_lines[i], col_width)}{rst}"
                lines.append(f"{l} {d}{div_char}{rst}")
        elif tag == "insert":
            for j in range(j1, j2):
                rv = f"{g}{truncate(right_lines[j], col_width)}{rst}"
                lines.append(f"{' ' * col_width} {d}{div_char}{rst} {rv}")
    return "\n".join(lines)


def render_unified_diff(prev_lines, curr_lines, prev_title, curr_title, pretty=True):
    diff = list(
        difflib.unified_diff(
            prev_lines,
            curr_lines,
            fromfile=prev_title,
            tofile=curr_title,
            lineterm="",
        )
    )
    if not pretty:
        return "\n".join(diff)

    colored = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            colored.append(f"{BOLD}{line}{RESET}")
        elif line.startswith("@@"):
            colored.append(f"{CYAN}{line}{RESET}")
        elif line.startswith("-"):
            colored.append(f"{RED}{line}{RESET}")
        elif line.startswith("+"):
            colored.append(f"{GREEN}{line}{RESET}")
        else:
            colored.append(f"{DIM}{line}{RESET}")
    return "\n".join(colored)
