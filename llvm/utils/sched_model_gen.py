import argparse
import json
import re

from fractions import Fraction


# A simple tool for generating a scheduling model draft from a SWOG.
#
# It depends on the following python packages:
#
#   pdf2docx
#
# Take "Neoverse N3 Core Software Optimization Guide.pdf" as an example. Below are
# steps to get a scheduling model draft:
#
# 1. Write an json file which contains the pipelines N3 has and how symbols in
#    instruction tables refer to them.
#
#    n3.json:
#
#    {
#        "name": "N3",
#        "pipelines": [
#            "Branch",
#            "Integer Single-Cycle",
#            "Integer Single/Multi-Cycle",
#            "FP/ASIMD/Vector Store data",
#            "Load/Store",
#            "Load 2",
#            "Integer Store data"
#        ],
#        "symbols": {
#              "B": ["Branch", 0, 1],
#              "S": ["Integer Single-Cycle", 0, 1],
#              "M": ["Integer Single/Multi-Cycle", 0, 1],
#              "I": ["Integer Single-Cycle", 0, 1, "Integer Single/Multi-Cycle", 0, 1],
#             "M0": ["Integer Single/Multi-Cycle", 0],
#            "L01": ["Load/Store", 0, 1],
#              "L": ["Load/Store", 0, 1, "Load 2", 0],
#             "ID": ["Integer Store data", 0, 1],
#              "V": ["FP/ASIMD/Vector Store data", 0, 1],
#             "V0": ["FP/ASIMD/Vector Store data", 0],
#             "V1": ["FP/ASIMD/Vector Store data", 1]
#        }
#    }
#
# 2. Generate the scheduling model draft.
#
#    python3 sched_model_gen.py -o n3.td n3.json "Neoverse N3 Core Software Optimization Guide.pdf"
#
# Basic idea:
#
# * We generate an InstRW for a row in the instruction tables based on a simple rule:
#   match the throughput assuming all utilized units are fully utilized.
#
#   Take the following row in the Neoverse N3 instruction tables as an example:
#
#   Instruction Group           Instructions  Latency  Throughput  Pipelines
#   Branch and link, register   BLR           1        2           B, S
#
#   The throughput is 2, that means 2 instructions are executed in a cycle.
#   The pipeline B and C each has 2 units, that means 2 B uops and 2 S uops
#   are executed in a cycle. So each instruction has 1 B uop and 1 S uop.
#


def extract(pdf_file: str):
    from pdf2docx import Converter

    cv = Converter(pdf_file)
    tables = cv.extract_tables()
    cv.close()

    def cook(s: str):
        if s is None:
            return ""
        s = s.replace("\n", " ").replace("\t", " ").strip()
        return s

    lines = []
    for table in tables:
        for row in table:
            lines.append("\t".join([cook(_) for _ in row]))
    return lines


def split(s, sep):
    s = s.split(sep)
    s = [_.strip() for _ in s]
    s = [_ for _ in s if _]
    return s


def join(arr, sep=""):
    return sep.join([str(_) for _ in arr])


def generate(rows: list[dict], o: dict, ofs):
    cpu_name: str = o["name"]

    pipelines = {}
    for x in o["pipelines"]:
        pipelines[x] = set()

    symbols = {}
    for name, arr in o["symbols"].items():
        desc = {}
        current = None
        for x in arr:
            if isinstance(x, str):
                assert x in pipelines
                assert x not in desc
                current = x
                desc[current] = []
            else:
                assert isinstance(x, int)
                assert current
                desc[current].append(x)

        symbols[name] = []
        for x in desc:
            desc[x] = tuple(sorted(set(desc[x])))
            pipelines[x].add(desc[x])
            symbols[name].append(tuple([x, desc[x]]))

    def divide(parts: list):
        while True:
            parts = [_ for _ in parts if _]

            def try_divide():
                n = len(parts)
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        pi = set(parts[i])
                        pj = set(parts[j])
                        pk = pi & pj
                        if pk:
                            parts[i] = tuple(sorted(pi - pk))
                            parts[j] = tuple(sorted(pj - pk))
                            parts.append(tuple(sorted(pk)))
                            return True
                return False

            if not try_divide():
                break
        return parts

    proc_resource_names = {}

    for name in pipelines:
        parts = list(pipelines[name])
        pipelines[name] = sorted(divide(parts))

        for x in pipelines[name]:
            proc_resource_names[tuple([name, x])] = ""

    for symbol_name, arr in symbols.items():
        if len(arr) == 1:
            t = arr[0]
            if t in proc_resource_names:
                proc_resource_names[t] = symbol_name

    for t in proc_resource_names:
        if not proc_resource_names[t]:
            name = t[0]
            s = "".join(
                [
                    _
                    for _ in name
                    if ord("a") <= ord(_) <= ord("z")
                    or ord("A") <= ord(_) <= ord("Z")
                    or ord("0") <= ord(_) <= ord("9")
                ]
            )
            if len(pipelines[name]) > 1:
                s += join(t[1])
            proc_resource_names[t] = s

    def _unit_name(x):
        return f"{cpu_name}Unit{x}"

    # Write ProcResource
    for name, arr in pipelines.items():
        for x in arr:
            resource_name = proc_resource_names[tuple([name, x])]

            comment = f"""{name} {join(x, '/')}"""
            if name[-1].isdigit():
                comment = name

            print(
                f"""def {_unit_name(resource_name)} : ProcResource<{len(x)}>; // {comment}""",
                file=ofs,
            )
    print("", file=ofs)

    def decompose(name, parts):
        res = []
        for x in pipelines[name]:
            if set(x) <= set(parts):
                res.append(x)
        return sorted(res)

    # Write ProcResGroup
    for symbol_name, arr in symbols.items():
        if len(arr) == 1 and arr[0] in proc_resource_names:
            continue

        names = []
        for t in arr:
            parts = decompose(t[0], t[1])
            for x in parts:
                names.append(proc_resource_names[tuple([t[0], x])])
        names = [_unit_name(_) for _ in names]
        print(
            f"""def {_unit_name(symbol_name)} : ProcResGroup<[{join(names, ', ')}]>;""",
            file=ofs,
        )

    writes = {}
    key_to_num_micro_ops = {}

    def add_write(latency, units, release_at_cycles, num_micro_ops, iterative: bool):
        assert len(units) == len(release_at_cycles)

        if iterative:
            num_micro_ops = len(release_at_cycles)

            postfix = "_".join(["1%s" % _ for _ in units])
            postfix = "%s_%s" % (postfix, join(release_at_cycles, "_"))

        else:
            units_2 = []
            release_at_cycles_2 = []
            for x, y in zip(units, release_at_cycles):
                units_2.extend([x] * y)
                release_at_cycles_2.extend([1] * y)

            postfix = "_".join(
                ["%s%s" % (_2, _1) for _1, _2 in zip(units, release_at_cycles)]
            )

            units = units_2
            release_at_cycles = release_at_cycles_2

        units = [f"{_unit_name(_)}" for _ in units]

        key = f"{cpu_name}Write_%sc_%s" % (latency, postfix)
        if key in writes:
            return key

        assert key not in key_to_num_micro_ops
        key_to_num_micro_ops[key] = num_micro_ops

        text = f"""
def {key} : SchedWriteRes<[{', '.join(units)}]> {{
    let Latency = {latency};"""

        if num_micro_ops > 1:
            text += f"""
    let NumMicroOps = {num_micro_ops};"""

        if sum(release_at_cycles) != len(release_at_cycles):
            text += f"""
    let ReleaseAtCycles = {release_at_cycles};"""

        text += """
}
"""
        writes[key] = text

        return key

    for row in rows:
        latency = row["latency"]
        throughput = row["throughput"]

        def _get_n_units(name):
            res = 0
            for x in symbols[name]:
                res += len(x[1])
            return res

        n_units = sum([_get_n_units(_) for _ in row["units"]])
        num_micro_ops = int(n_units / throughput)

        release_at_cycles = [int(_get_n_units(_) / throughput) for _ in row["units"]]

        if num_micro_ops != sum(release_at_cycles):
            print("[UNEXPECTED]", json.dumps(row))

        row["key"] = add_write(
            latency, row["units"], release_at_cycles, num_micro_ops, row["iterative"]
        )

    # Write SchedWriteRes
    for key in sorted(
        key_to_num_micro_ops.keys(), key=lambda _: key_to_num_micro_ops[_]
    ):
        print(writes[key], end="", file=ofs)

    # Write InstRW
    for row in rows:
        if "key" not in row:
            continue

        print(
            f"""
// {row['group']}
def : InstRW<[{row['key']}], (instrs {', '.join(row['instrs'])})>;
""",
            end="",
            file=ofs,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", metavar="TD_FILE", required=True)
    parser.add_argument("json_file", metavar="JSON_FILE")
    parser.add_argument("pdf_file", metavar="PDF_FILE")

    args = parser.parse_args()

    td_file = args.o
    json_file = args.json_file
    pdf_file = args.pdf_file

    lines = extract(pdf_file)

    with open(json_file, encoding="UTF-8") as ifs:
        o = json.load(ifs)

    def _get_row(line: list[str]):
        if len(line) == 5:
            line.append("")
        if len(line) != 6:
            return None

        latency = line[2]
        throughput = line[3]

        units = split(line[4], ",")
        if len(units) == 0:
            return None
        for x in units:
            if x not in o["symbols"]:
                return None

        if latency in ["-", ""]:
            print("[SKIPPED]", line)
            return None

        iterative = False

        if "to" in latency:
            latency = split(latency, "to")[1]
            iterative = True
        m = re.fullmatch(r"(\d+) ?\(\d+\)", latency)
        if m:
            latency = m.group(1)
        latency = int(latency)

        if "to" in throughput:
            throughput = split(throughput, "to")[0]
            iterative = True
        if "/" in throughput:
            throughput = Fraction(throughput)
        else:
            throughput = int(throughput)

        row = {
            "group": line[0],
            "instrs": split(line[1], ","),
            "latency": latency,
            "throughput": throughput,
            "units": units,
            "note": line[5],
            "iterative": iterative,
        }
        return row

    def get_row(line):
        try:
            return _get_row(line)
        except ValueError:
            return None

    def get_rows(lines: list[str]):
        rows = []
        state = "BEGIN"
        for line in lines:
            line = [_.strip() for _ in line.split("\t")]

            def is_head():
                return (
                    len(line) == 6
                    and line[0] == "Instruction Group"
                    and line[2] == "Exec Latency"
                    and line[3] == "Execution Throughput"
                )

            if state == "BEGIN":
                if is_head():
                    state = "ROW"
            else:
                assert state == "ROW"

                row = get_row(line)
                if row:
                    rows.append(row)
                else:
                    state = "ROW" if is_head() else "BEGIN"
        return rows

    rows = get_rows(lines)

    with open(td_file, "w", encoding="UTF-8") as ofs:
        generate(rows, o, ofs)


if __name__ == "__main__":
    main()
