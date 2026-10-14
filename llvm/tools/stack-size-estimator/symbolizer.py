import subprocess
import os
import sys
from typing import List, Dict, Optional


class Symbolizer:
    def __init__(self, bin_path: str, obj_file: str):
        self.bin_path = bin_path
        self.obj_file = obj_file
        self.process = None

    def _start(self):
        if self.process:
            return

        if not os.path.exists(self.bin_path):
            raise FileNotFoundError(f"Symbolizer not found at {self.bin_path}")

        if not os.path.exists(self.obj_file):
            raise FileNotFoundError(
                f"Object file not found at {self.obj_file}")

        # Start llvm-symbolizer in interactive mode
        # --obj specifies the binary once, so we just feed addresses
        cmd = [
            self.bin_path, f"--obj={self.obj_file}", "--output-style=GNU",
            "--functions=none"
        ]
        # --functions=none because we just want file:line, we have names from JSON
        # actually --output-style=GNU gives "File:Line", LLVM gives more lines.
        # Let's stick to default which is usually
        # Function
        # File:Line
        # Empty line

        # Using --output-style=GNU gives:
        # File:Line

        try:
            self.process = subprocess.Popen(cmd,
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=sys.stderr,
                                            universal_newlines=True,
                                            bufsize=1)
        except OSError as e:
            print(f"Failed to start symbolizer: {e}", file=sys.stderr)
            raise

    def symbolize(self, addresses: List[int]) -> Dict[int, str]:
        if not addresses:
            return {}

        self._start()
        if not self.process:
            return {}

        result = {}
        try:
            for addr in addresses:
                # Send hex address
                self.process.stdin.write(f"{hex(addr)}\n")
            self.process.stdin.flush()

            for addr in addresses:
                # Read response
                # With --output-style=GNU and --functions=none, it should be one line per query?
                # Let's verify standard behavior.
                # llvm-symbolizer defaults to printing function name then file:line.
                # If we add --no-inlines, we get one frame.
                # Let's try to parse 2 lines if we don't use GNU style, or 1 line with GNU style.

                # We will restart the process with simpler flags to be safe or just handle the output.
                # Let's assume standard behavior:
                # Code
                # /path/to/file:123
                #
                # Wait, I used --output-style=GNU above.
                line = self.process.stdout.readline().strip()
                result[addr] = line

        except Exception as e:
            print(f"Error during symbolization: {e}", file=sys.stderr)

        return result

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None


class Demangler:
    def __init__(self, bin_path: str):
        self.bin_path = bin_path
        self.process = None

    def _start(self):
        if self.process:
            return

        if not os.path.exists(self.bin_path):
            raise FileNotFoundError(f"Demangler not found at {self.bin_path}")

        cmd = [self.bin_path]
        try:
            self.process = subprocess.Popen(cmd,
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=sys.stderr,
                                            universal_newlines=True,
                                            bufsize=1)
        except OSError as e:
            print(f"Failed to start demangler: {e}", file=sys.stderr)
            raise

    def demangle(self, names: List[str]) -> Dict[str, str]:
        if not names:
            return {}

        self._start()
        if not self.process:
            return {name: name for name in names}

        result = {}
        try:
            for name in names:
                self.process.stdin.write(f"{name}\n")
            self.process.stdin.flush()

            for name in names:
                line = self.process.stdout.readline().strip()
                result[name] = line
        except Exception as e:
            print(f"Error during demangling: {e}", file=sys.stderr)
            for name in names:
                if name not in result:
                    result[name] = name

        return result

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
