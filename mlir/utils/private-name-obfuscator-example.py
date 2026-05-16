#!/usr/bin/env python3
"""Example obfuscator for ``MLIR_PRIVATE_NAME_OBFUSCATOR``.

This script reads a single identifier from stdin and prints a stable
integer index for it to stdout. The mapping is persisted in a hard-coded
table file: line ``N`` of the table holds the original name whose
obfuscated form is ``N``.

The table file *is* the obfuscation dictionary. To translate an
obfuscated identifier like ``_42`` that appears in a customer-reported
error message back to a human-readable name, simply read line 42 of the
table file with any text editor.

mlir-tblgen invokes this script as

    printf '%s' '<name>' | private-name-obfuscator-example.py

It then takes the first whitespace-delimited token of stdout (the integer
we print here), prefixes it with ``_``, and uses the result as the
obfuscated name. So if this script prints ``42``, mlir-tblgen substitutes
``_42`` for the input identifier.

Configure with::

    cmake -DMLIR_PRIVATE_NAME_OBFUSCATOR="<path-to-this-script>" ...

The script is *not* shipped or built as part of MLIR; it lives here only
as a runnable reference for downstream builds that want to bootstrap
their own obfuscator. Production deployments typically want to:

* pick a stable, location-independent table path (e.g. inside the
  source tree of the project being built),
* include the resulting table file in their release artifacts so the
  mapping is available for de-obfuscating customer reports,
* and possibly mix in a build-private salt or HMAC if the mapping itself
  needs to be kept secret.
"""

import fcntl
import os
import sys

# Hard-coded path of the obfuscation table. Replace with whatever location
# makes sense for your build.
TABLE_PATH = os.path.expanduser("~/.mlir-private-names.txt")


def main() -> int:
    name = sys.stdin.read()
    if not name:
        sys.stderr.write(
            "private-name-obfuscator-example: empty name on stdin\n"
        )
        return 1
    if "\n" in name:
        sys.stderr.write(
            "private-name-obfuscator-example: name contains a newline\n"
        )
        return 1

    # Ensure the table file exists before we open it for read+write.
    open(TABLE_PATH, "a").close()

    # Use an exclusive file lock so that concurrent mlir-tblgen invocations
    # (e.g. parallel ninja jobs) cannot race when extending the table.
    with open(TABLE_PATH, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        lines = f.read().splitlines()
        for index, existing in enumerate(lines, start=1):
            if existing == name:
                print(index)
                return 0
        # Not present yet: append and return the new (1-based) line number.
        new_index = len(lines) + 1
        f.write(name + "\n")
        print(new_index)
        return 0


if __name__ == "__main__":
    sys.exit(main())
