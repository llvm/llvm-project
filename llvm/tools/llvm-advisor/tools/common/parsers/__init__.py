# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from .base_parser import BaseParser
from .remarks_parser import RemarksParser
from .time_trace_parser import TimeTraceParser
from .diagnostics_parser import DiagnosticsParser
from .ast_parser import ASTParser
from .pgo_profile_parser import PGOProfileParser
from .xray_parser import XRayParser
from .static_analyzer_parser import StaticAnalyzerParser
from .ir_parser import IRParser
from .objdump_parser import ObjdumpParser
from .include_tree_parser import IncludeTreeParser
from .assembly_parser import AssemblyParser
from .preprocessed_parser import PreprocessedParser
from .sarif_parser import SARIFParser
from .macro_expansion_parser import MacroExpansionParser
from .dependencies_parser import DependenciesParser
from .binary_size_parser import BinarySizeParser
from .debug_parser import DebugParser
from .symbols_parser import SymbolsParser
from .runtime_trace_parser import RuntimeTraceParser
from .compilation_phases_parser import CompilationPhasesParser
from .ftime_report_parser import FTimeReportParser
from .version_info_parser import VersionInfoParser

__all__ = [
    "BaseParser",
    "RemarksParser",
    "TimeTraceParser",
    "DiagnosticsParser",
    "ASTParser",
    "PGOProfileParser",
    "XRayParser",
    "StaticAnalyzerParser",
    "IRParser",
    "ObjdumpParser",
    "IncludeTreeParser",
    "AssemblyParser",
    "PreprocessedParser",
    "SARIFParser",
    "MacroExpansionParser",
    "DependenciesParser",
    "BinarySizeParser",
    "DebugParser",
    "SymbolsParser",
    "RuntimeTraceParser",
    "CompilationPhasesParser",
    "FTimeReportParser",
    "VersionInfoParser",
]
