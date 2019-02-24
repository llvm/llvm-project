//===-- SurgicalInstrumentationConfig.cpp -- Surgical CSI -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of CSI, a framework that provides comprehensive static
// instrumentation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/SurgicalInstrumentationConfig.h"

namespace llvm {
InstrumentationPoint
ParseInstrumentationPoint(const StringRef &instrPointString) {
  if (SurgicalInstrumentationPoints.find(instrPointString) ==
      SurgicalInstrumentationPoints.end()) {
    return InstrumentationPoint::INSTR_INVALID_POINT;
  } else
    return SurgicalInstrumentationPoints[instrPointString];
}

std::unique_ptr<InstrumentationConfig>
llvm::InstrumentationConfig::GetDefault() {
  return std::unique_ptr<DefaultInstrumentationConfig>(
      new DefaultInstrumentationConfig());
}

std::unique_ptr<InstrumentationConfig>
InstrumentationConfig::ReadFromConfigurationFile(const std::string &filename) {
  auto file = MemoryBuffer::getFile(filename);

  if (!file) {
    llvm::report_fatal_error(
        "Instrumentation configuration file could not be opened: " +
        file.getError().message());
  }

  StringRef contents = file.get()->getBuffer();
  SmallVector<StringRef, 20> lines;

  contents.split(lines, '\n', -1, false);

  StringMap<InstrumentationPoint> functions;
  StringSet<> interposedFunctions;

  bool interposeMode = false;

  // One instruction per line.
  for (auto &line : lines) {
    auto trimmedLine = line.trim();
    if (trimmedLine.size() == 0 ||
        trimmedLine[0] == '#') // Skip comments or empty lines.
      continue;

    if (trimmedLine == "INTERPOSE") {
      interposeMode = true;
      continue;
    } else if (trimmedLine == "INSTRUMENT") {
      interposeMode = false;
      continue;
    }

    SmallVector<StringRef, 5> tokens;
    trimmedLine.split(tokens, ',', -1, false);

    if (interposeMode) {
      interposedFunctions.insert(tokens[0]);
    } else {
      if (tokens.size() > 0) {
        InstrumentationPoint points = InstrumentationPoint::INSTR_INVALID_POINT;
        if (tokens.size() >
            1) // This function specifies specific instrumentation points.
        {
          for (size_t i = 1; i < tokens.size(); ++i) {
            auto instrPoint = ParseInstrumentationPoint(tokens[i].trim());

            points |= instrPoint;
          }
        }

        auto trimmed = tokens[0].trim();
        if (trimmed != "")
          functions[trimmed] = points;
      }
    }
  }

  // If the configuration file turned out to be empty,
  // instrument everything.
  if (functions.size() == 0 && interposedFunctions.size() == 0)
    return GetDefault();

  for (auto &function : functions) {
    if (interposedFunctions.find(function.getKey()) != interposedFunctions.end()) {
      llvm::errs() << "warning: function for which interpositioning was "
                      "requested is also listed for instrumentation. The "
                      "function will only be interposed";
    }
  }

  return std::unique_ptr<InstrumentationConfig>(
      new InstrumentationConfig(functions, interposedFunctions));
}

} // namespace llvm
