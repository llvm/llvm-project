; RUN: opt -S -passes=debugify,globalopt -f %s | FileCheck %s

@foo = internal global i32 0, align 4

define void @store() {
entry:
  store i32 5, i32* @foo, align 4
  ret void
}

define i32 @bar() {
entry:
  %0 = load i32, i32* @foo, align 4
  ret i32 %0
}

;CHECK:      @bar
;CHECK-NEXT: entry:
;CHECK-NEXT:   %.b = load i1, i1* @foo, align 1, !dbg ![[DbgLocLoadSel:[0-9]+]]
;CHECK-NEXT:   %0 = select i1 %.b, i32 5, i32 0, !dbg ![[DbgLocLoadSel]]
;CHECK-NEXT:   call void @llvm.dbg.value({{.*}}), !dbg ![[DbgLocLoadSel]]
;CHECK-NEXT:   ret i32 %0, !dbg ![[DbgLocRet:[0-9]+]]

;CHECK: ![[DbgLocLoadSel]] = !DILocation(line: 3,
;CHECK: ![[DbgLocRet]] = !DILocation(line: 4,
