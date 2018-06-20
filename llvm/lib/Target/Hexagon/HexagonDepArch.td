//===- HexagonDepArch.td --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Automatically generated file, please consult code owner before editing.
//===----------------------------------------------------------------------===//


def ArchV65: SubtargetFeature<"v65", "HexagonArchVersion", "Hexagon::ArchEnum::V65", "Enable Hexagon V65 architecture">;
def HasV65 : Predicate<"HST->hasV65Ops()">, AssemblerPredicate<"ArchV65">;
def ArchV62: SubtargetFeature<"v62", "HexagonArchVersion", "Hexagon::ArchEnum::V62", "Enable Hexagon V62 architecture">;
def HasV62 : Predicate<"HST->hasV62Ops()">, AssemblerPredicate<"ArchV62">;
def ArchV60: SubtargetFeature<"v60", "HexagonArchVersion", "Hexagon::ArchEnum::V60", "Enable Hexagon V60 architecture">;
def HasV60 : Predicate<"HST->hasV60Ops()">, AssemblerPredicate<"ArchV60">;
def ArchV55: SubtargetFeature<"v55", "HexagonArchVersion", "Hexagon::ArchEnum::V55", "Enable Hexagon V55 architecture">;
def HasV55 : Predicate<"HST->hasV55Ops()">, AssemblerPredicate<"ArchV55">;
def ArchV4: SubtargetFeature<"v4", "HexagonArchVersion", "Hexagon::ArchEnum::V4", "Enable Hexagon V4 architecture">;
def HasV4 : Predicate<"HST->hasV4Ops()">, AssemblerPredicate<"ArchV4">;
def ArchV5: SubtargetFeature<"v5", "HexagonArchVersion", "Hexagon::ArchEnum::V5", "Enable Hexagon V5 architecture">;
def HasV5 : Predicate<"HST->hasV5Ops()">, AssemblerPredicate<"ArchV5">;
