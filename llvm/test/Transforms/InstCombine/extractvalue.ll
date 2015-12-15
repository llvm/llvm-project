; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @bar({i32, i32} %a)
declare i32 @baz(i32 %a)

; CHECK-LABEL: define i32 @foo(
; CHECK-NOT: extractvalue
define i32 @foo(i32 %a, i32 %b) {
; Instcombine should fold various combinations of insertvalue and extractvalue
; together
        ; Build a simple struct and pull values out again
        %s1.1 = insertvalue {i32, i32} undef, i32 %a, 0
        %s1 = insertvalue {i32, i32} %s1.1, i32 %b, 1
        %v1 = extractvalue {i32, i32} %s1, 0
        %v2 = extractvalue {i32, i32} %s1, 1

        ; Build a nested struct and pull a sub struct out of it
        ; This requires instcombine to insert a few insertvalue instructions
        %ns1.1 = insertvalue {i32, {i32, i32}} undef, i32 %v1, 0
        %ns1.2 = insertvalue {i32, {i32, i32}} %ns1.1, i32 %v1, 1, 0
        %ns1   = insertvalue {i32, {i32, i32}} %ns1.2, i32 %v2, 1, 1
        %s2    = extractvalue {i32, {i32, i32}} %ns1, 1
        %v3    = extractvalue {i32, {i32, i32}} %ns1, 1, 1
        call void @bar({i32, i32} %s2)

        ; Use nested extractvalues to get to a value
        %s3    = extractvalue {i32, {i32, i32}} %ns1, 1
        %v4    = extractvalue {i32, i32} %s3, 1
        call void @bar({i32, i32} %s3)

        ; Use nested insertvalues to build a nested struct
        %s4.1 = insertvalue {i32, i32} undef, i32 %v3, 0
        %s4   = insertvalue {i32, i32} %s4.1, i32 %v4, 1
        %ns2  = insertvalue {i32, {i32, i32}} undef, {i32, i32} %s4, 1

        ; And now extract a single value from there
        %v5   = extractvalue {i32, {i32, i32}} %ns2, 1, 1

        ret i32 %v5
}

; CHECK-LABEL: define i32 @extract2gep(
; CHECK-NEXT: [[GEP:%[a-z0-9]+]] = getelementptr inbounds {{.*}}, {{.*}}* %pair, i64 0, i32 1
; CHECK-NEXT: [[LOAD:%[A-Za-z0-9]+]] = load i32, i32* [[GEP]]
; CHECK-NEXT: store
; CHECK-NEXT: br label %loop
; CHECK-NOT: extractvalue
; CHECK: call {{.*}}(i32 [[LOAD]])
; CHECK-NOT: extractvalue
; CHECK: ret i32 [[LOAD]]
define i32 @extract2gep({i16, i32}* %pair, i32* %P) {
        ; The load + extractvalue should be converted
        ; to an inbounds gep + smaller load.
        ; The new load should be in the same spot as the old load.
        %L = load {i16, i32}, {i16, i32}* %pair
        store i32 0, i32* %P
        br label %loop

loop:
        %E = extractvalue {i16, i32} %L, 1
        %C = call i32 @baz(i32 %E)
        store i32 %C, i32* %P
        %cond = icmp eq i32 %C, 0
        br i1 %cond, label %end, label %loop

end:
        ret i32 %E
}

; CHECK-LABEL: define i16 @doubleextract2gep(
; CHECK-NEXT: [[GEP:%[a-z0-9]+]] = getelementptr inbounds {{.*}}, {{.*}}* %arg, i64 0, i32 1, i32 1
; CHECK-NEXT: [[LOAD:%[A-Za-z0-9]+]] = load i16, i16* [[GEP]]
; CHECK-NEXT: ret i16 [[LOAD]]
define i16 @doubleextract2gep({i16, {i32, i16}}* %arg) {
        ; The load + extractvalues should be converted
        ; to a 3-index inbounds gep + smaller load.
        %L = load {i16, {i32, i16}}, {i16, {i32, i16}}* %arg
        %E1 = extractvalue {i16, {i32, i16}} %L, 1
        %E2 = extractvalue {i32, i16} %E1, 1
        ret i16 %E2
}

; CHECK: define i32 @nogep-multiuse
; CHECK-NEXT: load {{.*}} %pair
; CHECK-NEXT: extractvalue
; CHECK-NEXT: extractvalue
; CHECK-NEXT: add
; CHECK-NEXT: ret
define i32 @nogep-multiuse({i32, i32}* %pair) {
        ; The load should be left unchanged since both parts are needed.
        %L = load volatile {i32, i32}, {i32, i32}* %pair
        %LHS = extractvalue {i32, i32} %L, 0
        %RHS = extractvalue {i32, i32} %L, 1
        %R = add i32 %LHS, %RHS
        ret i32 %R
}

; CHECK: define i32 @nogep-volatile
; CHECK-NEXT: load volatile {{.*}} %pair
; CHECK-NEXT: extractvalue
; CHECK-NEXT: ret
define i32 @nogep-volatile({i32, i32}* %pair) {
        ; The load volatile should be left unchanged.
        %L = load volatile {i32, i32}, {i32, i32}* %pair
        %E = extractvalue {i32, i32} %L, 1
        ret i32 %E
}
