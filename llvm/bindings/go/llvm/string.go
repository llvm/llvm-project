//===- string.go - Stringer implementation for Type -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stringer interface for the Type type.
//
//===----------------------------------------------------------------------===//

package llvm

import "fmt"

func (t TypeKind) String() string {
	switch t {
	case VoidTypeKind:
		return "VoidTypeKind"
	case FloatTypeKind:
		return "FloatTypeKind"
	case DoubleTypeKind:
		return "DoubleTypeKind"
	case X86_FP80TypeKind:
		return "X86_FP80TypeKind"
	case FP128TypeKind:
		return "FP128TypeKind"
	case PPC_FP128TypeKind:
		return "PPC_FP128TypeKind"
	case LabelTypeKind:
		return "LabelTypeKind"
	case IntegerTypeKind:
		return "IntegerTypeKind"
	case FunctionTypeKind:
		return "FunctionTypeKind"
	case StructTypeKind:
		return "StructTypeKind"
	case ArrayTypeKind:
		return "ArrayTypeKind"
	case PointerTypeKind:
		return "PointerTypeKind"
	case VectorTypeKind:
		return "VectorTypeKind"
	case MetadataTypeKind:
		return "MetadataTypeKind"
	}
	panic("unreachable")
}

func (t Type) String() string {
	ts := typeStringer{s: make(map[Type]string)}
	return ts.typeString(t)
}

type typeStringer struct {
	s map[Type]string
}

func (ts *typeStringer) typeString(t Type) string {
	if s, ok := ts.s[t]; ok {
		return s
	}

	k := t.TypeKind()
	s := k.String()
	s = s[:len(s)-len("Kind")]

	switch k {
	case ArrayTypeKind:
		s += fmt.Sprintf("(%v[%v])", ts.typeString(t.ElementType()), t.ArrayLength())
	case PointerTypeKind:
		s += fmt.Sprintf("(%v)", ts.typeString(t.ElementType()))
	case FunctionTypeKind:
		params := t.ParamTypes()
		s += "("
		if len(params) > 0 {
			s += fmt.Sprintf("%v", ts.typeString(params[0]))
			for i := 1; i < len(params); i++ {
				s += fmt.Sprintf(", %v", ts.typeString(params[i]))
			}
		}
		s += fmt.Sprintf("):%v", ts.typeString(t.ReturnType()))
	case StructTypeKind:
		if name := t.StructName(); name != "" {
			ts.s[t] = "%" + name
			s = fmt.Sprintf("%%%s: %s", name, s)
		}
		etypes := t.StructElementTypes()
		s += "("
		if n := len(etypes); n > 0 {
			s += ts.typeString(etypes[0])
			for i := 1; i < n; i++ {
				s += fmt.Sprintf(", %v", ts.typeString(etypes[i]))
			}
		}
		s += ")"
	case IntegerTypeKind:
		s += fmt.Sprintf("(%d bits)", t.IntTypeWidth())
	}

	ts.s[t] = s
	return s
}
