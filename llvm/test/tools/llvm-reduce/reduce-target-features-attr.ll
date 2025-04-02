; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=target-features-attr --test FileCheck --test-arg -enable-var-scope --test-arg --check-prefixes=INTERESTING,CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT,CHECK %s < %t

; CHECK: @keep_none_from_one() [[$KEEP_NONE_FROM_ONE:#[0-9]+]]
define void @keep_none_from_one() #0 {
  ret void
}

; CHECK: @keep_one_from_one() [[$KEEP_ONE_FROM_ONE:#[0-9]+]]
define void @keep_one_from_one() #1 {
  ret void
}

; CHECK: @keep_first_from_two() [[$KEEP_FIRST_FROM_TWO:#[0-9]+]]
define void @keep_first_from_two() #2 {
  ret void
}

; CHECK: @keep_second_from_two() [[$KEEP_SECOND_FROM_TWO:#[0-9]+]]
define void @keep_second_from_two() #3 {
  ret void
}

; CHECK: @keep_all_of_two() [[$KEEP_ALL_OF_TWO:#[0-9]+]]
define void @keep_all_of_two() #4 {
  ret void
}

; CHECK: @drop_empty_element() [[$DROP_EMPTY_ELEMENT:#[0-9]+]]
define void @drop_empty_element() #5 {
  ret void
}

; CHECK: @keep_second_from_three() [[$KEEP_SECOND_FROM_THREE:#[0-9]+]]
define void @keep_second_from_three() #6 {
  ret void
}

; RESULT: define void @no_target_features() {
define void @no_target_features() {
  ret void
}

; IR verifier should probably reject this
; RESULT: define void @no_target_features_value() {
define void @no_target_features_value() #7 {
  ret void
}

attributes #0 = { "target-features"="+foo" "unique-attr-0" }
attributes #1 = { "target-features"="+foo" "unique-attr-1" }
attributes #2 = { "target-features"="+first,+second" "unique-attr-2" }
attributes #3 = { "target-features"="+first,+second" "unique-attr-3" }
attributes #4 = { "target-features"="+first,+second" "unique-attr-4" }
attributes #5 = { "target-features"="+dead,,+beef" "unique-attr-5" }
attributes #6 = { "target-features"="+a,+b,+c" "unique-attr-6" }
attributes #7 = { "target-features" }

; INTERESTING-DAG: [[$KEEP_ONE_FROM_ONE]] = { "target-features"="+foo"
; INTERESTING-DAG: [[$KEEP_FIRST_FROM_TWO]] = { "target-features"="{{.*}}+first
; INTERESTING-DAG: [[$KEEP_SECOND_FROM_TWO]] = { "target-features"="{{.*}}+second
; INTERESTING-DAG: [[$KEEP_ALL_OF_TWO]] = { "target-features"="{{.*}}+first,+second
; INTERESTING-DAG: [[$DROP_EMPTY_ELEMENT]] = { "target-features"="{{.*}}+dead{{.*}}+beef
; INTERESTING-DAG: [[$KEEP_SECOND_FROM_THREE]] = { "target-features"="{{.*}}+b


; RESULT-DAG: attributes [[$KEEP_NONE_FROM_ONE]] = { "unique-attr-0" }
; RESULT-DAG: [[$KEEP_FIRST_FROM_TWO]] = { "target-features"="+first" "unique-attr-2" }
; RESULT-DAG: [[$KEEP_SECOND_FROM_TWO]] = { "target-features"="+second" "unique-attr-3" }
; RESULT-DAG: [[$KEEP_ALL_OF_TWO]] = { "target-features"="+first,+second" "unique-attr-4" }
; RESULT-DAG: [[$DROP_EMPTY_ELEMENT]] = { "target-features"="+dead,+beef" "unique-attr-5" }
; RESULT-DAG: [[$KEEP_SECOND_FROM_THREE]] = { "target-features"="+b" "unique-attr-6" }
