# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from typing import Dict, List, Any
from .base_parser import BaseParser
from ..models import FileType, ParsedFile, Dependency


class DependenciesParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.DEPENDENCIES)

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            dependencies = self._parse_dependencies(lines)

            # Calculate statistics
            sources = set()
            targets = set()
            for dep in dependencies:
                sources.add(dep.source)
                targets.add(dep.target)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_dependencies": len(dependencies),
                "unique_sources": len(sources),
                "unique_targets": len(targets),
                "unique_files": len(sources.union(targets)),
            }

            return self.create_parsed_file(file_path, dependencies, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_dependencies(self, lines: List[str]) -> List[Dependency]:
        dependencies = []
        current_target = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Handle make-style dependencies (target: source1 source2 ...)
            if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    target = parts[0].strip()
                    sources = parts[1].strip()
                    current_target = target

                    if sources:
                        for source in sources.split():
                            source = source.strip()
                            if source and source != "\\":
                                dependencies.append(
                                    Dependency(
                                        source=source, target=target, type="dependency"
                                    )
                                )

            # Handle continuation lines
            elif (line.startswith(" ") or line.startswith("\t")) and current_target:
                sources = line.strip()
                for source in sources.split():
                    source = source.strip()
                    if source and source != "\\":
                        dependencies.append(
                            Dependency(
                                source=source, target=current_target, type="dependency"
                            )
                        )

            # Handle simple dependency lists (one per line)
            elif "->" in line or "=>" in line:
                if "->" in line:
                    parts = line.split("->", 1)
                else:
                    parts = line.split("=>", 1)

                if len(parts) == 2:
                    source = parts[0].strip()
                    target = parts[1].strip()
                    dependencies.append(
                        Dependency(source=source, target=target, type="dependency")
                    )

            # Reset current target for new sections
            elif not line.startswith(" ") and not line.startswith("\t"):
                current_target = None

        return dependencies
