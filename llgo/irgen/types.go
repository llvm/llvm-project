//===- types.go - convenience functions for types -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements convenience functions for dealing with types.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"llvm.org/llgo/third_party/gotools/go/types"
)

func deref(t types.Type) types.Type {
	return t.Underlying().(*types.Pointer).Elem()
}
