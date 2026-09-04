; RUN: opt -passes=debugify --debugify-atoms -S -o - < %s \
; RUN: | FileCheck %s

;; Mirrors llvm/test/DebugInfo/debugify.ll

; CHECK-LABEL: define void @foo
define void @foo() {
; CHECK: ret void, !dbg ![[RET1:.*]]
  ret void
}

; CHECK-LABEL: define i32 @bar
define i32 @bar() {
; CHECK: call void @foo(), !dbg ![[CALL1:.*]]
  call void @foo()

; CHECK: add i32 0, 1, !dbg ![[ADD1:.*]]
  %sum = add i32 0, 1

; CHECK: ret i32 0, !dbg ![[RET2:.*]]
  ret i32 0
}

; CHECK-LABEL: define weak_odr zeroext i1 @baz
define weak_odr zeroext i1 @baz() {
; CHECK-NOT: !dbg
  ret i1 false
}

; CHECK-LABEL: define i32 @boom
define i32 @boom() {
; CHECK: [[result:%.*]] = musttail call i32 @bar(), !dbg ![[musttail:.*]]
  %retval = musttail call i32 @bar()
; CHECK-NEXT: ret i32 [[result]], !dbg ![[musttailRes:.*]]
  ret i32 %retval
}

; CHECK: distinct !DISubprogram(name: "foo", {{.*}}keyInstructions: true)
; CHECK-DAG: ![[RET1]] = !DILocation(line: 1, {{.*}}, atomGroup: 1, atomRank: 1
; CHECK: distinct !DISubprogram(name: "bar", {{.*}}keyInstructions: true)
; CHECK-DAG: ![[CALL1]] = !DILocation(line: 2, {{.*}}, atomGroup: 2, atomRank: 1
; CHECK-DAG: ![[ADD1]] = !DILocation(line: 3, {{.*}}, atomGroup: 3, atomRank: 1
; CHECK-DAG: ![[RET2]] = !DILocation(line: 4, {{.*}}, atomGroup: 4, atomRank: 1
; CHECK: distinct !DISubprogram(name: "boom", {{.*}}keyInstructions: true)
; CHECK-DAG: ![[musttail]] = !DILocation(line: 5, {{.*}}, atomGroup: 5, atomRank: 1
; CHECK-DAG: ![[musttailRes]] = !DILocation(line: 6, {{.*}}, atomGroup: 6, atomRank: 1
