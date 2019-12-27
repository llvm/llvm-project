; RUN: llc -march=hexagon -mcpu=hexagonv62 -mtriple=hexagon-unknown-linux-musl < %s | FileCheck %s
; CHECK-LABEL: PrintInts:
; CHECK-DAG: memw{{.*}} = r{{[0-9]+}}
; CHECK-DAG: memw{{.*}} = r{{[0-9]+}}
; CHECK-DAG: r{{[0-9]+}}:{{[0-9]+}} = memd{{.*}}
; CHECK-DAG: memd{{.*}} = r{{[0-9]+}}:{{[0-9]+}}

%struct.__va_list_tag = type { i8*, i8*, i8* }

; Function Attrs: nounwind
define void @PrintInts(i32 %first, ...) #0 {
entry:
  %vl = alloca [1 x %struct.__va_list_tag], align 8
  %vl_count = alloca [1 x %struct.__va_list_tag], align 8
  %arraydecay1 = bitcast [1 x %struct.__va_list_tag]* %vl to i8*
  call void @llvm.va_start(i8* %arraydecay1)
  %0 = bitcast [1 x %struct.__va_list_tag]* %vl_count to i8*
  call void @llvm.va_copy(i8* %0, i8* %arraydecay1)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) #1

; Function Attrs: nounwind
declare void @llvm.va_copy(i8*, i8*) #1

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  tail call void (i32, ...) @PrintInts(i32 undef, i32 20, i32 30, i32 40, i32 50, i32 0)
  ret i32 0
}

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
