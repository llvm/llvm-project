; RUN: not opt --foo 2>&1 | grep "Unknown command line argument"

; there is no --foo
