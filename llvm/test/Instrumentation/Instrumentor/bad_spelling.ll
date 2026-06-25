; RUN: opt < %s -passes=instrumentor -instrumentor-read-config-files=%S/bad_spelling_config.json -S 2>&1 | FileCheck %s

; CHECK: warning: malformed JSON configuration, expected an object matching an instrumentor choice, got 'numerci'; did you mean 'numeric'?
; CHECK-NEXT: warning: configuration key 'gpu_enabeld' not found and ignored; did you mean 'gpu_enabled'?
; CHECK-NEXT: warning: unrecognized JSON property 'opocde' in configuration for 'numeric'; did you mean 'opcode'?
; CHECK-NEXT: warning: unrecognized JSON property 'zzzzzz' in configuration for 'numeric'
; CHECK-NEXT: warning: unrecognized JSON property 'enabld' in configuration for 'numeric'; did you mean 'enabled'?