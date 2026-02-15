; REQUIRES: default_triple

;-------------------------------------------------------------------------------
; default 
;-------------------------------------------------------------------------------
; RUN: rm -rf %t/logs
; RUN: llc %s -o - -O3 -print-after-all -print-before-all \
; RUN:     -ir-dump-directory %t/logs
; RUN: %python %p/check-ir-dump-filenames.py %t/logs \
; RUN:     "(?P<pass_number>[0-9]+)" \
; RUN:     "-(?P<kind>module|function|machine-function)" \
; RUN:     "-(?P<pass_id>.+)" \
; RUN:     "-(?P<suffix>before|after)"

; RUN: rm -rf %t/logs
; RUN: opt %s -disable-output -O3 -print-after-all -print-before-all \
; RUN:     -ir-dump-directory %t/logs
; RUN: %python %p/check-ir-dump-filenames.py %t/logs \
; RUN:     "(?P<pass_number>[0-9]+)" \
; RUN:     "-(?P<kind>[0-9a-f]+-(module|((function|scc)-[0-9a-f]+)))" \
; RUN:     "-(?P<pass_id>.+)" \
; RUN:     "-(?P<suffix>before|after)"

; RUN: rm -rf %t/logs
; RUN: llc %s -o - -O3 -print-before-all \
; RUN:     -ir-dump-directory %t/logs
; RUN: %python %p/check-ir-dump-filenames.py %t/logs \
; RUN:     "(?P<pass_number>[0-9]+)" \
; RUN:     "-(?P<kind>module|function|machine-function)" \
; RUN:     "-(?P<pass_id>.+)" \
; RUN:     "-(?P<before>before)"

; RUN: rm -rf %t/logs
; RUN: opt %s -disable-output -O3 -print-after-all \
; RUN:     -ir-dump-directory %t/logs
; RUN: %python %p/check-ir-dump-filenames.py %t/logs \
; RUN:     "(?P<pass_number>[0-9]+)" \
; RUN:     "-(?P<kind>[0-9a-f]+-(module|((function|scc)-[0-9a-f]+)))" \
; RUN:     "-(?P<pass_id>.+)" \
; RUN:     "-(?P<after>after)"

;-------------------------------------------------------------------------------
; sortable
;-------------------------------------------------------------------------------
; RUN: rm -rf %t/logs
; RUN: llc %s -o - -O3 -print-after-all -print-before-all \
; RUN:     -ir-dump-directory %t/logs \
; RUN:     -ir-dump-filename-format sortable
; RUN: %python %p/check-ir-dump-filenames.py %t/logs \
; RUN:     "(?P<ordinal>[0-9]{8})" \
; RUN:     "-(?P<pass_ordinal>[0-9]{8})" \
; RUN:     "-(?P<kind>module|function|machine-function)" \
; RUN:     "-(?P<pass_id>.+)" \
; RUN:     "-(?P<suffix_ordinal>[01])" \
; RUN:     "-(?P<suffix>before|after)"

; RUN: rm -rf %t/logs
; RUN: opt %s -disable-output -O3 -print-after-all -print-before-all \
; RUN:     -ir-dump-directory %t/logs \
; RUN:     -ir-dump-filename-format sortable
; RUN: %python %p/check-ir-dump-filenames.py %t/logs \
; RUN:     "(?P<ordinal>[0-9]{8})" \
; RUN:     "-(?P<pass_number>[0-9]+)" \
; RUN:     "-(?P<kind>[0-9a-f]+-(module|((function|scc)-[0-9a-f]+)))" \
; RUN:     "-(?P<pass_id>.+)" \
; RUN:     "-(?P<suffix_ordinal>[01])" \
; RUN:     "-(?P<suffix>before|after)"

;-------------------------------------------------------------------------------
; thread id
;-------------------------------------------------------------------------------
; RUN: rm -rf %t/logs
; RUN: llc %s -o - -O3 -print-after-all \
; RUN:     -ir-dump-directory %t/logs \
; RUN:     -ir-dump-filename-prepend-thread-id
; RUN: %python %p/check-ir-dump-filenames.py %t/logs \
; RUN:     "(?P<tid>[0-9]+)" \
; RUN:     "-(?P<pass_number>[0-9]+)" \
; RUN:     "-(?P<kind>module|function|machine-function)" \
; RUN:     "-(?P<pass_id>.+)" \
; RUN:     "-(?P<after>after)"

define void @foo() {
    ret void
}
