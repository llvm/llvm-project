; SI run line skipped since store not yet implemented.
; RUN: llc -global-isel -march=amdgcn -mcpu=tonga -verify-machineinstrs < %S/../readcyclecounter.ll | FileCheck -enable-var-scope -check-prefix=MEMTIME -check-prefix=SIVI -check-prefix=GCN %S/../readcyclecounter.ll
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %S/../readcyclecounter.ll | FileCheck -enable-var-scope -check-prefix=MEMTIME -check-prefix=GCN %S/../readcyclecounter.ll
