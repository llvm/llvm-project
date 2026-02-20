# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import json
from typing import List, Dict, Any
from .base_parser import BaseParser
from ..models import FileType, ParsedFile, Diagnostic, SourceLocation


class SARIFParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.STATIC_ANALYSIS_SARIF)

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            sarif_data = json.loads(content)
            diagnostics = []

            # Parse SARIF format
            runs = sarif_data.get("runs", [])
            for run in runs:
                results = run.get("results", [])
                for result in results:
                    diagnostic = self._parse_sarif_result(result, run)
                    if diagnostic:
                        diagnostics.append(diagnostic)

            metadata = {
                "total_results": len(diagnostics),
                "file_size": self.get_file_size(file_path),
                "sarif_version": sarif_data.get("$schema", ""),
                "runs_count": len(runs),
            }

            return self.create_parsed_file(file_path, diagnostics, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_sarif_result(
        self, result: Dict[str, Any], run: Dict[str, Any]
    ) -> Diagnostic:
        try:
            message = result.get("message", {}).get("text", "")
            rule_id = result.get("ruleId", "")

            # Extract level from result
            level = result.get("level", "info")

            # Extract location
            location = None
            locations = result.get("locations", [])
            if locations:
                physical_location = locations[0].get("physicalLocation", {})
                artifact_location = physical_location.get("artifactLocation", {})
                region = physical_location.get("region", {})

                if artifact_location.get("uri"):
                    location = SourceLocation(
                        file=artifact_location.get("uri"),
                        line=region.get("startLine"),
                        column=region.get("startColumn"),
                    )

            return Diagnostic(
                level=level, message=message, location=location, code=rule_id
            )

        except Exception:
            return None
