; RUN: opt -mtriple=amdgcn -passes="require<runtime-libcall-info>,expand-fp<O0>" -disable-output %s
; RUN: opt -mtriple=amdgcn -passes="require<runtime-libcall-info>,expand-fp<O1>" -disable-output %s
; RUN: opt -mtriple=amdgcn -passes="require<runtime-libcall-info>,expand-fp<O2>" -disable-output %s
; RUN: opt -mtriple=amdgcn -passes="require<runtime-libcall-info>,expand-fp<O3>" -disable-output %s

; RUN: not opt -mtriple=amdgcn -passes="expand-fp<O4>" %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=TOO-LARGE %s
; TOO-LARGE: {{.*}}invalid optimization level for expand-fp pass: 4

; RUN: not opt -mtriple=amdgcn -passes="expand-fp<Os>" %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=NON-NUMERIC %s
; NON-NUMERIC: {{.*}}invalid expand-fp pass parameter

; RUN: not opt -mtriple=amdgcn -passes="expand-fp<O-1>" %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=NEGATIVE %s
; NEGATIVE: {{.*}}invalid expand-fp pass parameter 'O-1'

; RUN: not opt -mtriple=amdgcn -passes="expand-fp<foo>" %s -S -o /dev/null 2>&1 | FileCheck --check-prefix=NO-O-PREFIX %s
; NO-O-PREFIX: {{.*}}invalid expand-fp pass parameter 'foo'

define void @empty() {
  ret void
}
