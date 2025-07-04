REQUIRES: aarch64-registered-target

RUN: llvm-exegesis -mcpu=neoverse-v2 --benchmark-phase=measure --min-instructions=100 --mode=latency --print-gen-assembly=10 --opcode-name=ADDVv4i16v 2>&1 | FileCheck %s -check-prefix=PREVIEW

PREVIEW: Generated assembly snippet:
PREVIEW-NEXT: ```
PREVIEW:      {{[04]}}: {{.*}}      movi    d{{[0-9]+}}, #0000000000000000
PREVIEW-NEXT: {{[48]}}: {{.*}}      addv    h{{[0-9]+}}, v{{[0-9]+}}.4h
PREVIEW:      ...      ({{[0-9]+}} more instructions)
PREVIEW-NEXT: {{.*}}   addv    h{{[0-9]+}}, v{{[0-9]+}}.4h
PREVIEW:      {{.*}}   ret
PREVIEW-NEXT:```