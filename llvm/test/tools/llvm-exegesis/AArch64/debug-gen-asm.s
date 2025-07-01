REQUIRES: aarch64-registered-target

RUN: llvm-exegesis -mcpu=neoverse-v2 --benchmark-phase=measure --min-instructions=100 --mode=latency --debug-only="preview-gen-assembly" --opcode-name=ADDVv4i16v 2>&1 | FileCheck %s -check-prefix=PREVIEW

PREVIEW: Generated assembly snippet:
PREVIEW-NEXT: ```
PREVIEW:      {{.*}}   movi    d{{[0-9]+}}, #0000000000000000
PREVIEW-NEXT: {{.*}}   addv    h{{[0-9]+}}, v{{[0-9]+}}.4h
PREVIEW:      {{.*}}   addv    h{{[0-9]+}}, v{{[0-9]+}}.4h
PREVIEW:      ...      ({{[0-9]+}} more instructions)
PREVIEW-NEXT: {{.*}}   addv    h{{[0-9]+}}, v{{[0-9]+}}.4h
PREVIEW:      {{.*}}   ret
PREVIEW-NEXT:```

RUN: llvm-exegesis -mcpu=neoverse-v2 --benchmark-phase=measure --min-instructions=100 --mode=latency --debug-only="print-gen-assembly" --opcode-name=ADDVv4i16v 2>&1 | FileCheck %s -check-prefix=PRINT

PRINT: Generated assembly snippet:
PRINT-NEXT: ```
PRINT:       {{.*}}     movi    d{{[0-9]+}}, #0000000000000000
PRINT:       {{.*}}     addv    h{{[0-9]+}}, v{{[0-9]+}}.4h
PRINT:       {{.*}}     ret
PRINT-NEXT: ```
