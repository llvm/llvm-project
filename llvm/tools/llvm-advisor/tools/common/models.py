# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum


class FileType(Enum):
    REMARKS = "remarks"
    TIME_TRACE = "time-trace"
    DIAGNOSTICS = "diagnostics"
    AST_JSON = "ast-json"
    PGO_PROFILE = "pgo-profile"
    XRAY = "xray"
    STATIC_ANALYZER = "static-analyzer"
    IR = "ir"
    OBJDUMP = "objdump"
    INCLUDE_TREE = "include-tree"
    ASSEMBLY = "assembly"
    PREPROCESSED = "preprocessed"
    STATIC_ANALYSIS_SARIF = "static-analysis-sarif"
    MACRO_EXPANSION = "macro-expansion"
    DEPENDENCIES = "dependencies"
    BINARY_SIZE = "binary-size"
    DEBUG = "debug"
    SYMBOLS = "symbols"
    RUNTIME_TRACE = "runtime-trace"
    COMPILATION_PHASES = "compilation-phases"
    FTIME_REPORT = "ftime-report"
    VERSION_INFO = "version-info"
    SOURCES = "sources"


@dataclass
class SourceLocation:
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass
class CompilationUnit:
    name: str
    path: str
    artifacts: Dict[FileType, List[str]]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ParsedFile:
    file_type: FileType
    file_path: str
    data: Any
    metadata: Dict[str, Any]


@dataclass
class Diagnostic:
    level: str
    message: str
    location: Optional[SourceLocation] = None
    code: Optional[str] = None


@dataclass
class Remark:
    pass_name: str
    function: str
    message: str
    location: Optional[SourceLocation] = None
    args: Dict[str, Any] = None


@dataclass
class TraceEvent:
    name: str
    category: str
    phase: str
    timestamp: int
    duration: Optional[int] = None
    pid: Optional[int] = None
    tid: Optional[int] = None
    args: Dict[str, Any] = None


@dataclass
class Symbol:
    name: str
    address: Optional[str] = None
    size: Optional[int] = None
    type: Optional[str] = None
    section: Optional[str] = None


@dataclass
class BinarySize:
    section: str
    size: int
    percentage: Optional[float] = None


@dataclass
class Dependency:
    source: str
    target: str
    type: Optional[str] = None
