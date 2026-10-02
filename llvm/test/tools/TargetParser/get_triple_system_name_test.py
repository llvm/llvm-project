# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
"""Tests for get_triple_system_name.

The parser must stay in sync with TripleName.def, and system_name() must
classify the same OS that Triple::normalize picks, including for the
unnormalized/legacy triple shapes the runtimes build may feed in.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "utils"))

import get_triple_system_name as g


class Test(unittest.TestCase):
    def setUp(self):
        self.names = g.parse(g.find_def_file())

    def test_all_def_rows_parsed(self):
        # Guard against the regex silently dropping a .def row the C++ accepts.
        with open(g.find_def_file()) as f:
            lines = [l for l in f if "#define" not in l]

        def count(macro):
            return sum(1 for l in lines if l.lstrip().startswith(macro + "("))

        self.assertEqual(
            len(self.names.os), count("TRIPLE_OS") + count("TRIPLE_OS_ALIAS")
        )
        self.assertEqual(len(self.names.env), count("TRIPLE_ENV"))

    def test_prefix_ordering(self):
        # A name that is a prefix of a LATER name would shadow it under the
        # StartsWith matching both the C++ and this script rely on.
        for table in (self.names.os, self.names.env):
            names = [n for n, _ in table]
            for i, short in enumerate(names):
                for longer in names[i + 1 :]:
                    self.assertFalse(
                        longer.startswith(short) and longer != short,
                        f"'{short}' precedes and shadows '{longer}'",
                    )

    def test_every_os_row(self):
        # Data-driven: a canonical arch-unknown-<os> triple must map to the
        # CMake name the .def declares for every OS (and alias) row. This is
        # the exhaustive counterpart to the hand-picked cases below; it fails
        # automatically when a new OS is added without a CMake mapping.
        for name, cmake_name in self.names.os:
            # firmware/driverkit/bridgeos also have vendor-specific behavior
            # exercised separately; here the plain arch-unknown form suffices.
            triple = f"arch-unknown-{name}"
            self.assertEqual(g.system_name(self.names, triple), cmake_name, triple)

    def test_known_triples(self):
        cases = {
            "x86_64-unknown-linux-gnu": "Linux",
            "arm64-apple-darwin": "Darwin",
            "s390x-ibm-zos": "OS390",
            "aarch64-unknown-linux-android": "Android",
            "x86_64-pc-windows-cygnus": "CYGWIN",
            "wasm32-unknown-wasip1": "WASI",
            "arm64-apple-ios": "iOS",
            "riscv64-unknown-elf": "Generic",
        }
        for triple, want in cases.items():
            self.assertEqual(g.system_name(self.names, triple), want, triple)

    def test_os_aliases(self):
        # The TRIPLE_OS_ALIAS rows resolve to their target OS's CMake name.
        cases = {
            "x86_64-apple-macos": "Darwin",  # -> MacOSX
            "x86_64-pc-win32": "Windows",  # -> Win32
            "arm64-apple-visionos": "visionOS",  # -> XROS
        }
        for triple, want in cases.items():
            self.assertEqual(g.system_name(self.names, triple), want, triple)

    def test_non_canonical_triples(self):
        # Vendor omitted / non-standard, os or env not in the fixed position,
        # version suffixes, and GCC-legacy spellings.
        cases = {
            "aarch64-linux-android21": "Android",
            "aarch64-unknown-linux-android21": "Android",
            "arm-linux-androideabi": "Android",
            "armv7a-linux-androideabi29": "Android",
            # Vendor omitted, os in the vendor slot.
            "x86_64-linux-gnu": "Linux",
            "riscv64-linux-gnu": "Linux",
            # OS version suffix.
            "x86_64-apple-macosx10.15": "Darwin",
            "armv7-apple-ios13.0": "iOS",
            # Non-standard vendor, still resolvable by os component.
            "x86_64-pc-freebsd14": "FreeBSD",
        }
        for triple, want in cases.items():
            self.assertEqual(g.system_name(self.names, triple), want, triple)

    def test_matches_normalize_funky_triples(self):
        # The "real-world funky triples" from TripleTest.cpp's Normalization
        # test. system_name must classify the same OS that Triple::normalize
        # picks, including os-in-vendor-slot and GCC-legacy windows spellings
        # (mingw* -> windows-gnu, cygwin*/msys -> windows-cygnus).
        cases = {
            "i386-mingw32": "Windows",  # -> i386-unknown-windows-gnu
            "x86_64-linux-gnu": "Linux",  # -> x86_64-unknown-linux-gnu
            "i486-linux-gnu": "Linux",  # -> i486-unknown-linux-gnu
            "i386-redhat-linux": "Linux",  # -> i386-redhat-linux
            "i686-linux": "Linux",  # -> i686-unknown-linux
            "arm-none-eabi": "Generic",  # -> arm-unknown-none-eabi (no OS)
            "ve-linux": "Linux",  # -> ve-unknown-linux
            "wasm32-wasi": "WASI",  # -> wasm32-unknown-wasi
            "wasm64-wasi": "WASI",  # -> wasm64-unknown-wasi
            "x86_64-pc-cygwin": "CYGWIN",  # -> x86_64-pc-windows-cygnus
            "x86_64-pc-msys": "CYGWIN",  # -> x86_64-pc-windows-cygnus
            "x86_64-w64-mingw32": "Windows",  # -> x86_64-w64-windows-gnu
            "i686-w64-mingw32": "Windows",
        }
        for triple, want in cases.items():
            self.assertEqual(g.system_name(self.names, triple), want, triple)

    def test_normalize_special_cases(self):
        # The OS/environment special-cases that Triple::normalize applies, one
        # per category, to ensure the script tracks each rewrite it performs.
        cases = {
            # Win32 normalizes to "windows" regardless of the incoming spelling
            # or msvc/gnu environment.
            "x86_64-pc-windows": "Windows",
            "x86_64-pc-windows-msvc": "Windows",
            "x86_64-pc-windows-gnu": "Windows",
            # mingw* -> windows-gnu; cygwin*/msys -> windows-cygnus.
            "x86_64-w64-mingw32": "Windows",
            "x86_64-pc-cygwin": "CYGWIN",
            "x86_64-pc-msys": "CYGWIN",
            # androideabi keeps the Android environment override.
            "arm-linux-androideabi": "Android",
            # The full Apple OS family. driverkit/bridgeos map to Darwin
            # directly in the table (they are always Apple); firmware has no
            # dedicated CMake name and maps to Darwin via the apple-vendor
            # catch-all.
            "x86_64-apple-macosx": "Darwin",
            "arm-apple-darwin": "Darwin",
            "arm-apple-ios": "iOS",
            "arm-apple-tvos": "tvOS",
            "arm-apple-watchos": "watchOS",
            "arm-apple-xros": "visionOS",
            "arm64-apple-visionos": "visionOS",
            "arm-apple-driverkit": "Darwin",
            "arm-apple-bridgeos": "Darwin",
            "arm-apple-firmware": "Darwin",
        }
        for triple, want in cases.items():
            self.assertEqual(g.system_name(self.names, triple), want, triple)

    def test_firmware_requires_apple_vendor(self):
        # firmware has no dedicated CMake name and only maps to Darwin for the
        # apple vendor. With any other vendor it is Generic. (For a non-apple
        # vendor Triple::normalize actually fatal-errors; here we only need to
        # ensure we never claim Darwin.)
        for vendor in ("none", "unknown", "pc"):
            triple = f"arm-{vendor}-firmware"
            self.assertEqual(g.system_name(self.names, triple), "Generic", triple)

    def test_driverkit_bridgeos_always_darwin(self):
        # driverkit and bridgeos are always Apple, so they map to Darwin
        # directly in the table regardless of the vendor component.
        for os_name in ("driverkit", "bridgeos"):
            for vendor in ("apple", "none", "unknown", "pc"):
                triple = f"arm-{vendor}-{os_name}"
                self.assertEqual(g.system_name(self.names, triple), "Darwin", triple)

    def test_bare_os_no_arch(self):
        # A lone os component with no arch is a valid clang target (e.g.
        # --target=darwin -> unknown-unknown-darwin), so it must classify by
        # the first (and only) component rather than being skipped as the arch.
        cases = {
            "darwin": "Darwin",
            "linux": "Linux",
            "ios": "iOS",
            "freebsd": "FreeBSD",
            "wasi": "WASI",
            "zos": "OS390",
            "mingw32": "Windows",
            "cygwin": "CYGWIN",
        }
        for triple, want in cases.items():
            self.assertEqual(g.system_name(self.names, triple), want, triple)

    def test_no_classifiable_os_falls_back(self):
        # Only when no component classifies as an OS does the caller fall back
        # to the host system name.
        self.assertIsNone(g.system_name(self.names, "x86_64"))
        self.assertIsNone(g.system_name(self.names, "arm-none"))


if __name__ == "__main__":
    unittest.main()
