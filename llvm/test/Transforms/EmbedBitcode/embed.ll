; RUN: opt --mtriple x86_64-unknown-linux-gnu < %s -passes="embed-bitcode" -S | FileCheck %s
; RUN: opt --mtriple x86_64-unknown-linux-gnu < %s -passes="embed-bitcode<thinlto>" -S | FileCheck %s
; RUN: opt --mtriple x86_64-unknown-linux-gnu < %s -passes="embed-bitcode<emit-summary>" -S | FileCheck %s
; RUN: opt --mtriple x86_64-unknown-linux-gnu < %s -passes="embed-bitcode<thinlto;emit-summary>" -S | FileCheck %s

@a = global i32 1

; CHECK: @a = global i32 1
;; Make sure the module is in the correct section.
; CHECK: @llvm.embedded.object = private constant {{.*}}, section ".llvm.lto", align 1

;; Ensure that the metadata is in llvm.compiler.used.
; CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @llvm.embedded.object], section "llvm.metadata"

;; Make sure the metadata correlates to the .llvm.lto section.
; CHECK: !llvm.embedded.objects = !{!1}
; CHECK: !0 = !{}
; CHECK: !{ptr @llvm.embedded.object, !".llvm.lto"}


;; Ensure that the .llvm.lto section has SHT_EXCLUDE set.
; RUN: opt --mtriple x86_64-unknown-linux-gnu < %s -passes="embed-bitcode<thinlto;emit-summary>" -S \
; RUN: | llc --mtriple x86_64-unknown-linux-gnu -filetype=obj \
; RUN: | llvm-readobj - --sections --elf-output-style=JSON --pretty-print \
; RUN: | FileCheck %s --check-prefix=EXCLUDE

; EXCLUDE:        "Name": ".llvm.lto",
; EXCLUDE-NEXT:   "Value": 7
; EXCLUDE-NEXT: },
; EXCLUDE-NEXT: "Type": {
; EXCLUDE-NEXT:   "Name": "SHT_PROGBITS",
; EXCLUDE-NEXT:   "Value": 1
; EXCLUDE-NEXT: },
; EXCLUDE-NEXT: "Flags": {
; EXCLUDE-NEXT:   "Value": 2147483648,
; EXCLUDE-NEXT:   "Flags": [
; EXCLUDE-NEXT:     {
; EXCLUDE-NEXT:       "Name": "SHF_EXCLUDE",
; EXCLUDE-NEXT:       "Value": 2147483648
; EXCLUDE-NEXT:     }
; EXCLUDE-NEXT:   ]
; EXCLUDE-NEXT: },

