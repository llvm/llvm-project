; RUN: llc -march=sparc --relocation-model=pic < %s | FileCheck %s --check-prefix=V8
; RUN: llc -march=sparcv9 --relocation-model=pic < %s | FileCheck %s --check-prefix=V9
; RUN: llc -march=sparc   --relocation-model=pic < %s -O0 | FileCheck %s --check-prefix=V8UNOPT
; RUN: llc -march=sparcv9 --relocation-model=pic < %s -O0 | FileCheck %s --check-prefix=V9UNOPT


; V8-LABEL: func
; V8:  _GLOBAL_OFFSET_TABLE_

; V9-LABEL: func
; V9:  _GLOBAL_OFFSET_TABLE_

@foo = global i32 0                               ; <i32*> [#uses=1]

define i32 @func(i32 %a) nounwind readonly {
entry:
  %0 = load i32, i32* @foo, align 4                    ; <i32> [#uses=1]
  ret i32 %0
}

; V8UNOPT-LABEL: test_spill
; V8UNOPT:       sethi %hi(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[R:%[goli][0-7]]]
; V8UNOPT:       or [[R]], %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[R]]
; V8UNOPT:       add [[R]], %o7, [[R]]
; V8UNOPT:       st [[R]], [%fp+{{.+}}]

; V9UNOPT-LABEL: test_spill
; V9UNOPT:       sethi %hi(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[R:%[goli][0-7]]]
; V9UNOPT:       or [[R]], %lo(_GLOBAL_OFFSET_TABLE_+{{.+}}), [[R]]
; V9UNOPT:       add [[R]], %o7, [[R]]
; V9UNOPT:       stx [[R]], [%fp+{{.+}}]

define i32 @test_spill(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %ret =  load i32, i32* @foo, align 4
  ret i32 %ret

if.end:
  %add = add nsw i32 %b, %a
  ret i32 %add
}
