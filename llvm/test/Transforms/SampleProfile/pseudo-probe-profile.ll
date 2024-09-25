; RUN: opt < %s -passes=pseudo-probe,sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-profile.prof -pass-remarks=sample-profile -pass-remarks-output=%t.opt.yaml -sample-profile-use-profi=0 -S -o %t
; RUN: FileCheck %s --input-file %t
; RUN: FileCheck %s -check-prefix=YAML --input-file %t.opt.yaml
; RUN: opt < %t -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-profile.prof -sample-profile-remove-probe -S | FileCheck %s -check-prefix=REMOVE-PROBE

; REMOVE-PROBE-NOT: call void @llvm.pseudoprobe

define dso_local i32 @foo(i32 %x, ptr %f) #0 !dbg !4 {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %cmp = icmp eq i32 %0, 0
  ; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0, i64 -1)
  br i1 %cmp, label %if.then, label %if.else
  ; CHECK: br i1 %cmp, label %if.then, label %if.else, !prof ![[PD1:[0-9]+]]

if.then:
  ; CHECK: call {{.*}}, !dbg ![[#PROBE1:]], !prof ![[PROF1:[0-9]+]]
  call void %f(i32 1)
  ; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 2, i32 0, i64 -1)
  store i32 1, ptr %retval, align 4
  br label %return

if.else:
  ; CHECK: call {{.*}}, !dbg ![[#PROBE2:]], !prof ![[PROF2:[0-9]+]]
  call void %f(i32 2)
  ; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 4, i32 0, i64 -1)
  store i32 2, ptr %retval, align 4
  br label %return

return:
  ; CHECK: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 6, i32 0, i64 -1)
  %1 = load i32, ptr %retval, align 4
  ret i32 %1
}

attributes #0 = {"use-sample-profile"}

; CHECK: ![[PD1]] = !{!"branch_weights", i32 8, i32 7}
; CHECK: ![[#PROBE1]] = !DILocation(line: 0, scope: ![[#SCOPE1:]])
;; A discriminator of 387973151 which is 0x1720001f in hexdecimal, stands for an indirect call probe
;; with an index of 3.
; CHECK: ![[#SCOPE1]] = !DILexicalBlockFile(scope: ![[#]], file: ![[#]], discriminator: 387973151)
; CHECK: ![[PROF1]] = !{!"VP", i32 0, i64 7, i64 9191153033785521275, i64 5, i64 -1069303473483922844, i64 2}
;; A discriminator of 387973167 which is 0x1720002f in hexdecimal, stands for an indirect call probe
;; with an index of 5.
; CHECK: ![[#PROBE2]] = !DILocation(line: 0, scope: ![[#SCOPE2:]])
; CHECK: ![[#SCOPE2]] = !DILexicalBlockFile(scope: ![[#]], file: ![[#]], discriminator: 387973167)
; CHECK: ![[PROF2]] = !{!"VP", i32 0, i64 6, i64 -1069303473483922844, i64 4, i64 9191153033785521275, i64 2}

!llvm.module.flags = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}

; Checking to see if YAML file is generated and contains remarks
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: test.c, Line: 0, Column: 0 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '13'
;YAML-NEXT:    - String:          ' samples from profile (ProbeId='
;YAML-NEXT:    - ProbeId:         '1'
;YAML-NEXT:    - String:          ', Factor='
;YAML-NEXT:    - Factor:          '1.000000e+00'
;YAML-NEXT:    - String:          ', OriginalSamples='
;YAML-NEXT:    - OriginalSamples: '13'
;YAML-NEXT:    - String:          ')'
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: test.c, Line: 0, Column: 0 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '7'
;YAML-NEXT:    - String:          ' samples from profile (ProbeId='
;YAML-NEXT:    - ProbeId:         '3'
;YAML-NEXT:    - String:          ', Factor='
;YAML-NEXT:    - Factor:          '1.000000e+00'
;YAML-NEXT:    - String:          ', OriginalSamples='
;YAML-NEXT:    - OriginalSamples: '7'
;YAML-NEXT:    - String:          ')'
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: test.c, Line: 0, Column: 0 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '7'
;YAML-NEXT:    - String:          ' samples from profile (ProbeId='
;YAML-NEXT:    - ProbeId:         '2'
;YAML-NEXT:    - String:          ', Factor='
;YAML-NEXT:    - Factor:          '1.000000e+00'
;YAML-NEXT:    - String:          ', OriginalSamples='
;YAML-NEXT:    - OriginalSamples: '7'
;YAML-NEXT:    - String:          ')'
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: test.c, Line: 0, Column: 0 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '6'
;YAML-NEXT:    - String:          ' samples from profile (ProbeId='
;YAML-NEXT:    - ProbeId:         '5'
;YAML-NEXT:    - String:          ', Factor='
;YAML-NEXT:    - Factor:          '1.000000e+00'
;YAML-NEXT:    - String:          ', OriginalSamples='
;YAML-NEXT:    - OriginalSamples: '6'
;YAML-NEXT:    - String:          ')'
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: test.c, Line: 0, Column: 0 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '6'
;YAML-NEXT:    - String:          ' samples from profile (ProbeId='
;YAML-NEXT:    - ProbeId:         '4'
;YAML-NEXT:    - String:          ', Factor='
;YAML-NEXT:    - Factor:          '1.000000e+00'
;YAML-NEXT:    - String:          ', OriginalSamples='
;YAML-NEXT:    - OriginalSamples: '6'
;YAML-NEXT:    - String:          ')'
;YAML:  --- !Analysis
;YAML-NEXT:  Pass:            sample-profile
;YAML-NEXT:  Name:            AppliedSamples
;YAML-NEXT:  DebugLoc:        { File: test.c, Line: 0, Column: 0 }
;YAML-NEXT:  Function:        foo
;YAML-NEXT:  Args:
;YAML-NEXT:    - String:          'Applied '
;YAML-NEXT:    - NumSamples:      '13'
;YAML-NEXT:    - String:          ' samples from profile (ProbeId='
;YAML-NEXT:    - ProbeId:         '6'
;YAML-NEXT:    - String:          ', Factor='
;YAML-NEXT:    - Factor:          '1.000000e+00'
;YAML-NEXT:    - String:          ', OriginalSamples='
;YAML-NEXT:    - OriginalSamples: '13'
;YAML-NEXT:    - String:          ')'
