//===--- HexagonDepArch.td ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def ArchV62: SubtargetFeature<"v62", "HexagonArchVersion", "Hexagon::ArchEnum::V62", "Enable Hexagon V62 architecture">;
def HasV62T : Predicate<"HST->hasV62TOps()">, AssemblerPredicate<"ArchV62">;
def ArchV60: SubtargetFeature<"v60", "HexagonArchVersion", "Hexagon::ArchEnum::V60", "Enable Hexagon V60 architecture">;
def HasV60T : Predicate<"HST->hasV60TOps()">, AssemblerPredicate<"ArchV60">;
def ArchV55: SubtargetFeature<"v55", "HexagonArchVersion", "Hexagon::ArchEnum::V55", "Enable Hexagon V55 architecture">;
def HasV55T : Predicate<"HST->hasV55TOps()">, AssemblerPredicate<"ArchV55">;
def ArchV4: SubtargetFeature<"v4", "HexagonArchVersion", "Hexagon::ArchEnum::V4", "Enable Hexagon V4 architecture">;
def HasV4T : Predicate<"HST->hasV4TOps()">, AssemblerPredicate<"ArchV4">;
def ArchV5: SubtargetFeature<"v5", "HexagonArchVersion", "Hexagon::ArchEnum::V5", "Enable Hexagon V5 architecture">;
def HasV5T : Predicate<"HST->hasV5TOps()">, AssemblerPredicate<"ArchV5">;
