; RUN: opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-ir-insts<O0>" %s -S -disable-output
; RUN: opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-ir-insts<O1>" %s -S -disable-output
; RUN: opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-ir-insts<O2>" %s -S -disable-output
; RUN: opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-ir-insts<O3>" %s -S -disable-output

; RUN: not opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-ir-insts<O4>" %s -S -disable-output 2>&1 | FileCheck --check-prefix=TOO-LARGE %s
; TOO-LARGE: {{.*}}invalid optimization level for expand-ir-insts pass: 4

; RUN: not opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-ir-insts<Os>" %s -S -disable-output 2>&1 | FileCheck --check-prefix=NON-NUMERIC %s
; NON-NUMERIC: {{.*}}invalid expand-ir-insts pass parameter

; RUN: not opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-ir-insts<O-1>" %s -S -disable-output 2>&1 | FileCheck --check-prefix=NEGATIVE %s
; NEGATIVE: {{.*}}invalid expand-ir-insts pass parameter 'O-1'

; RUN: not opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-ir-insts<foo>" %s -S -disable-output 2>&1 | FileCheck --check-prefix=NO-O-PREFIX %s
; NO-O-PREFIX: {{.*}}invalid expand-ir-insts pass parameter 'foo'

define void @empty() {
  ret void
}
