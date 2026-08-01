import argparse


# Generate enough distinct body records to exercise multi-byte payload sizes and
# partial buffering before the configured payload limit is exceeded.
parser = argparse.ArgumentParser(
    description="Generate a sample profile with a large typified LBR payload"
)
parser.add_argument("output")
parser.add_argument("records", type=int)
args = parser.parse_args()

assert 0 < args.records <= 65536

with open(args.output, "w", encoding="utf-8", newline="\n") as output:
    output.write(f"large:{args.records}:1\n")
    for line_offset in range(args.records):
        output.write(f" {line_offset}: 1\n")
