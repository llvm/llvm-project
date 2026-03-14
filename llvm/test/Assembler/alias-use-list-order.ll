; RUN: verify-uselistorder < %s

; Globals.
@global = global i32 0
@alias.ref1 = global ptr getelementptr inbounds (i32, ptr @alias, i64 1)
@alias.ref2 = global ptr getelementptr inbounds (i32, ptr @alias, i64 1)

; Aliases.
@alias = alias i32, ptr @global
@alias.ref3 = alias i32, getelementptr inbounds (i32, ptr @alias, i64 1)
@alias.ref4 = alias i32, getelementptr inbounds (i32, ptr @alias, i64 1)
