// RUN: mlir-irdl-to-cpp %s --verify-diagnostics
// expected-error@+1 {{Error in dialect name: No leading or double underscores allowed.}}
irdl.dialect @_no_leading_underscore {
}

// expected-error@+1 {{Error in dialect name: No leading or double underscores allowed.}}
irdl.dialect @_no__double__underscores {
}

// expected-error@+1 {{Error in dialect name: Upper-case characters are not allowed}}
irdl.dialect @NoUpperCase {
}

// expected-error@+1 {{Error in dialect name: Only numbers and lower-case characters allowed}}
irdl.dialect @no_weird_symbol$ {
}

// ----- 