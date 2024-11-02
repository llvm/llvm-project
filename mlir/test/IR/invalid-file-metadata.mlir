// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error@+2 {{expected identifier key in file metadata dictionary}}
{-#

// -----

// expected-error@+2 {{expected ':'}}
{-#
  key
#-}

// -----

// expected-error@+2 {{unknown key 'some_key' in file metadata dictionary}}
{-#
  some_key: {}
#-}

// -----

//===----------------------------------------------------------------------===//
// `dialect_resources`
//===----------------------------------------------------------------------===//

// expected-error@+2 {{expected '{'}}
{-#
  dialect_resources: "value"
#-}

// -----

// expected-error@+3 {{expected identifier key for 'resource' entry}}
{-#
  dialect_resources: {
    10
  }
#-}

// -----

// expected-error@+3 {{expected ':'}}
{-#
  dialect_resources: {
    entry "value"
  }
#-}

// -----

// expected-error@+3 {{dialect 'foobar' is unknown}}
{-#
  dialect_resources: {
    foobar: {
      entry: "foo"
    }
  }
#-}

// -----

// expected-error@+4 {{unknown 'resource' key 'unknown_entry' for dialect 'ml_program'}}
{-#
  dialect_resources: {
    ml_program: {
      unknown_entry: "foo"
    }
  }
#-}

// -----

// expected-error@+4 {{expected hex string blob for key 'invalid_blob'}}
{-#
  dialect_resources: {
    test: {
      invalid_blob: 10
    }
  }
#-}

// -----

// expected-error@+4 {{expected hex string blob for key 'invalid_blob'}}
{-#
  dialect_resources: {
    test: {
      invalid_blob: ""
    }
  }
#-}

// -----

// expected-error@+4 {{expected hex string blob for key 'invalid_blob' to encode alignment in first 4 bytes}}
{-#
  dialect_resources: {
    test: {
      invalid_blob: "0x"
    }
  }
#-}

// -----

//===----------------------------------------------------------------------===//
// `external_resources`
//===----------------------------------------------------------------------===//

// expected-error@+2 {{expected '{'}}
{-#
  external_resources: "value"
#-}

// -----

// expected-error@+3 {{expected identifier key for 'resource' entry}}
{-#
  external_resources: {
    10
  }
#-}

// -----

// expected-error@+3 {{expected ':'}}
{-#
  external_resources: {
    entry "value"
  }
#-}
