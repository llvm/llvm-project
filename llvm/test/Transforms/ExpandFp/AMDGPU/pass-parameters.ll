; RUN: opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-fp<O0>" %s -S -disable-output
; RUN: opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-fp<O1>" %s -S -disable-output
; RUN: opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-fp<O2>" %s -S -disable-output
; RUN: opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-fp<O3>" %s -S -disable-output

; RUN: not opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-fp<O4>" %s -S -disable-output 2>&1 | FileCheck --check-prefix=TOO-LARGE %s
; TOO-LARGE: {{.*}}invalid optimization level for expand-fp pass: 4

; RUN: not opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-fp<Os>" %s -S -disable-output 2>&1 | FileCheck --check-prefix=NON-NUMERIC %s
; NON-NUMERIC: {{.*}}invalid expand-fp pass parameter

; RUN: not opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-fp<O-1>" %s -S -disable-output 2>&1 | FileCheck --check-prefix=NEGATIVE %s
; NEGATIVE: {{.*}}invalid expand-fp pass parameter 'O-1'

; RUN: not opt -mtriple=amdgcn -passes="require<libcall-lowering-info>,expand-fp<foo>" %s -S -disable-output 2>&1 | FileCheck --check-prefix=NO-O-PREFIX %s
; NO-O-PREFIX: {{.*}}invalid expand-fp pass parameter 'foo'

define void @empty() {
  ret void
}
