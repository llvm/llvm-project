# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
#
# This is the artifact collector module. It provides logic for discovering and
# parsing build artifacts for LLVM Advisor analysis.
#
# ===----------------------------------------------------------------------===#

import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from .models import FileType, CompilationUnit, ParsedFile
from .parsers import (
    RemarksParser,
    TimeTraceParser,
    DiagnosticsParser,
    ASTParser,
    PGOProfileParser,
    XRayParser,
    StaticAnalyzerParser,
    IRParser,
    ObjdumpParser,
    IncludeTreeParser,
    AssemblyParser,
    PreprocessedParser,
    SARIFParser,
    MacroExpansionParser,
    DependenciesParser,
    BinarySizeParser,
    DebugParser,
    SymbolsParser,
    RuntimeTraceParser,
    CompilationPhasesParser,
    FTimeReportParser,
    VersionInfoParser,
    PreprocessedParser as SourcesParser,  # Reuse for simple text files
)


class ArtifactCollector:
    def __init__(self):
        self.parsers = {
            FileType.REMARKS: RemarksParser(),
            FileType.TIME_TRACE: TimeTraceParser(),
            FileType.DIAGNOSTICS: DiagnosticsParser(),
            FileType.AST_JSON: ASTParser(),
            FileType.PGO_PROFILE: PGOProfileParser(),
            FileType.XRAY: XRayParser(),
            FileType.STATIC_ANALYZER: StaticAnalyzerParser(),
            FileType.IR: IRParser(),
            FileType.OBJDUMP: ObjdumpParser(),
            FileType.INCLUDE_TREE: IncludeTreeParser(),
            FileType.ASSEMBLY: AssemblyParser(),
            FileType.PREPROCESSED: PreprocessedParser(),
            FileType.STATIC_ANALYSIS_SARIF: SARIFParser(),
            FileType.MACRO_EXPANSION: MacroExpansionParser(),
            FileType.DEPENDENCIES: DependenciesParser(),
            FileType.BINARY_SIZE: BinarySizeParser(),
            FileType.DEBUG: DebugParser(),
            FileType.SYMBOLS: SymbolsParser(),
            FileType.RUNTIME_TRACE: RuntimeTraceParser(),
            FileType.COMPILATION_PHASES: CompilationPhasesParser(),
            FileType.FTIME_REPORT: FTimeReportParser(),
            FileType.VERSION_INFO: VersionInfoParser(),
            FileType.SOURCES: SourcesParser(),
        }

        # Map directory names to file types
        self.dir_to_type = {
            "remarks": FileType.REMARKS,
            "time-trace": FileType.TIME_TRACE,
            "diagnostics": FileType.DIAGNOSTICS,
            "ast-json": FileType.AST_JSON,
            "pgo-profile": FileType.PGO_PROFILE,
            "xray": FileType.XRAY,
            "static-analyzer": FileType.STATIC_ANALYZER,
            "ir": FileType.IR,
            "objdump": FileType.OBJDUMP,
            "include-tree": FileType.INCLUDE_TREE,
            "assembly": FileType.ASSEMBLY,
            "preprocessed": FileType.PREPROCESSED,
            "static-analysis-sarif": FileType.STATIC_ANALYSIS_SARIF,
            "macro-expansion": FileType.MACRO_EXPANSION,
            "dependencies": FileType.DEPENDENCIES,
            "binary-size": FileType.BINARY_SIZE,
            "debug": FileType.DEBUG,
            "symbols": FileType.SYMBOLS,
            "runtime-trace": FileType.RUNTIME_TRACE,
            "compilation-phases": FileType.COMPILATION_PHASES,
            "ftime-report": FileType.FTIME_REPORT,
            "version-info": FileType.VERSION_INFO,
            "sources": FileType.SOURCES,
        }

    def discover_compilation_units(self, advisor_dir: str) -> List[CompilationUnit]:
        """Discover all compilation units in the .llvm-advisor directory."""
        compilation_units = []
        advisor_path = Path(advisor_dir)

        if not advisor_path.exists():
            return compilation_units

        # Each subdirectory represents a compilation unit
        for unit_dir in advisor_path.iterdir():
            if not unit_dir.is_dir():
                continue

            # Check if this is the new nested structure or old flat structure
            units = self._scan_compilation_unit_with_runs(unit_dir)
            compilation_units.extend(units)

        return compilation_units

    def _scan_compilation_unit_with_runs(self, unit_dir: Path) -> List[CompilationUnit]:
        """Scan a compilation unit directory that contains timestamped runs."""
        units = []

        # unit_dir contains timestamped run directories
        run_dirs = []
        for item in unit_dir.iterdir():
            if item.is_dir() and item.name.startswith(unit_dir.name + "_"):
                run_dirs.append(item)

        if not run_dirs:
            # No timestamped runs found, skip this unit
            return units

        # Sort by timestamp (newest first)
        run_dirs.sort(key=lambda x: x.name, reverse=True)

        # Use the most recent run
        latest_run = run_dirs[0]
        unit = self._scan_single_run(latest_run, unit_dir.name)
        if unit:
            # Store run timestamp info in metadata
            unit.metadata = getattr(unit, "metadata", {})
            unit.metadata["run_timestamp"] = latest_run.name.split("_", 1)[-1]
            unit.metadata["run_path"] = str(latest_run)
            unit.metadata["available_runs"] = [r.name for r in run_dirs]
            units.append(unit)

        return units

    def _scan_single_run(
        self, run_dir: Path, unit_name: str
    ) -> Optional[CompilationUnit]:
        """Scan a single run directory for artifacts."""
        artifacts = {}

        # Scan each artifact type directory
        for artifact_dir in run_dir.iterdir():
            if not artifact_dir.is_dir():
                continue

            dir_name = artifact_dir.name
            if dir_name not in self.dir_to_type:
                continue

            file_type = self.dir_to_type[dir_name]
            artifact_files = []

            # Collect all files in this artifact directory
            for file_path in artifact_dir.rglob("*"):
                if file_path.is_file():
                    artifact_files.append(str(file_path))

            if artifact_files:
                artifacts[file_type] = artifact_files

        if artifacts:
            return CompilationUnit(
                name=unit_name, path=str(run_dir), artifacts=artifacts
            )

        return None

    def parse_compilation_unit(
        self, unit: CompilationUnit
    ) -> Dict[FileType, List[ParsedFile]]:
        """Parse all artifacts for a compilation unit."""
        parsed_artifacts = {}

        for file_type, file_paths in unit.artifacts.items():
            if file_type not in self.parsers:
                continue

            parser = self.parsers[file_type]
            parsed_files = []

            for file_path in file_paths:
                try:
                    if parser.can_parse(file_path):
                        parsed_file = parser.parse(file_path)
                        parsed_files.append(parsed_file)
                except Exception as e:
                    # Create error entry for failed parsing
                    error_file = ParsedFile(
                        file_type=file_type,
                        file_path=file_path,
                        data={},
                        metadata={"error": f"Failed to parse: {str(e)}"},
                    )
                    parsed_files.append(error_file)

            if parsed_files:
                parsed_artifacts[file_type] = parsed_files

        return parsed_artifacts

    def parse_all_units(
        self, advisor_dir: str
    ) -> Dict[str, Dict[FileType, List[ParsedFile]]]:
        """Parse all compilation units in the advisor directory."""
        units = self.discover_compilation_units(advisor_dir)
        parsed_units = {}

        for unit in units:
            parsed_artifacts = self.parse_compilation_unit(unit)
            if parsed_artifacts:
                parsed_units[unit.name] = parsed_artifacts

        return parsed_units

    def get_summary_statistics(
        self, parsed_units: Dict[str, Dict[FileType, List[ParsedFile]]]
    ) -> Dict[str, Any]:
        """Generate summary statistics for all parsed data."""
        stats = {
            "total_units": len(parsed_units),
            "total_files": 0,
            "file_types": {},
            "errors": 0,
            "units": {},
        }

        for unit_name, artifacts in parsed_units.items():
            unit_stats = {"file_types": {}, "total_files": 0, "errors": 0}

            for file_type, parsed_files in artifacts.items():
                type_name = file_type.value
                file_count = len(parsed_files)
                error_count = sum(1 for f in parsed_files if "error" in f.metadata)

                unit_stats["file_types"][type_name] = {
                    "count": file_count,
                    "errors": error_count,
                }
                unit_stats["total_files"] += file_count
                unit_stats["errors"] += error_count

                # Update global stats
                if type_name not in stats["file_types"]:
                    stats["file_types"][type_name] = {"count": 0, "errors": 0}

                stats["file_types"][type_name]["count"] += file_count
                stats["file_types"][type_name]["errors"] += error_count

            stats["units"][unit_name] = unit_stats
            stats["total_files"] += unit_stats["total_files"]
            stats["errors"] += unit_stats["errors"]

        return stats
