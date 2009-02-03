; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep extractvalue

; Instcombine should fold various combinations of insertvalue and extractvalue
; together
declare void @bar({i32, i32} %a)

define i32 @foo() {
        ; Build a simple struct and pull values out again
        %s1.1 = insertvalue {i32, i32} undef, i32 0, 0
        %s1 = insertvalue {i32, i32} %s1.1, i32 1, 1
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

