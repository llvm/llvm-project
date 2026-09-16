import log_reader
import interactive_host
import sys

from typing import Sequence


def evict_advice(tensor_values: Sequence[log_reader.TensorValue]):
    # this advisor just picks the first legal register to evict, which is
    # identifiable by the "mask" feature
    for tv in tensor_values:
        if tv.spec().name != "mask":
            continue
        for i, v in enumerate(tv):
            if v == 1:
                return i
    # i.e. invalid:
    return -1


def li_size(tensor_values: Sequence[log_reader.TensorValue]) -> float:
    for tv in tensor_values:
        if tv.spec().name == "li_size":
            return float(tv[0])
    return 0.0


# A constant policy is indistinguishable from any other, because RAGreedy then
# breaks all the resulting ties on the vreg number.
def split_policy(large_advice: float):
    def advice(tvs):
        size = li_size(tvs)
        return large_advice if size > 100 else size

    return advice


PRIORITY_POLICIES = {
    "low": split_policy(0.0),
    "high": split_policy(4.0e9),
    "negative": split_policy(-1.0),
    "nan": split_policy(float("nan")),
    "huge": split_policy(1.0e30),
}


def main(args):
    if args[0].startswith("--priority="):
        name = args[0][len("--priority=") :]
        if name not in PRIORITY_POLICIES:
            sys.exit(
                f"unknown priority policy {name!r}, expected one of "
                f"{sorted(PRIORITY_POLICIES)}"
            )
        advice = PRIORITY_POLICIES[name]
        args = args[1:]
    else:
        advice = evict_advice

    interactive_host.run_interactive(args[0], advice, args[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
