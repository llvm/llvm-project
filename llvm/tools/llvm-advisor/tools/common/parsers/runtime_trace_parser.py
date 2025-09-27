# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from .time_trace_parser import TimeTraceParser
from ..models import FileType


class RuntimeTraceParser(TimeTraceParser):
    def __init__(self):
        # Runtime trace uses the same Chrome trace format as time-trace
        super().__init__()
        self.file_type = FileType.RUNTIME_TRACE
