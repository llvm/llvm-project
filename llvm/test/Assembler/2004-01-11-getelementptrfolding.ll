; RUN: llvm-as < %s | llvm-dis | \
; RUN:   not grep "getelementptr.*getelementptr"
; RUN: verify-uselistorder %s

%struct.TTriangleItem = type { ptr, ptr, [3 x %struct.TUVVertex] }
%struct.TUVVertex = type { i16, i16, i16, i16 }
@data_triangleItems = internal constant [2908 x %struct.TTriangleItem] zeroinitializer; <ptr> [#uses=2]

define void @foo() {
        store i16 0, ptr getelementptr ([2908 x %struct.TTriangleItem], ptr @data_triangleItems, i64 0, i64 0, i32 2, i64 0, i32 0)
        ret void
}

