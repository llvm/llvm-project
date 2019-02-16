//===- attribute.go - attribute processor ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file processes llgo and //extern attributes.
//
//===----------------------------------------------------------------------===//

package irgen

import (
	"fmt"
	"go/ast"
	"llvm.org/llvm/bindings/go/llvm"
	"strings"
)

const AttributeCommentPrefix = "#llgo "

// Attribute represents an attribute associated with a
// global variable or function.
type Attribute interface {
	Apply(llvm.Value)
}

// parseAttribute parses zero or more #llgo comment attributes associated with
// a global variable or function. The comment group provided will be processed
// one line at a time using parseAttribute.
func parseAttributes(doc *ast.CommentGroup) []Attribute {
	var attributes []Attribute
	if doc == nil {
		return attributes
	}
	for _, comment := range doc.List {
		if strings.HasPrefix(comment.Text, "//extern ") {
			nameattr := nameAttribute(strings.TrimSpace(comment.Text[9:]))
			attributes = append(attributes, nameattr)
			continue
		}
		text := comment.Text[2:]
		if strings.HasPrefix(comment.Text, "/*") {
			text = text[:len(text)-2]
		}
		attr := parseAttribute(strings.TrimSpace(text))
		if attr != nil {
			attributes = append(attributes, attr)
		}
	}
	return attributes
}

// parseAttribute parses a single #llgo comment attribute associated with
// a global variable or function. The string provided will be parsed
// if it begins with AttributeCommentPrefix, otherwise nil is returned.
func parseAttribute(line string) Attribute {
	if !strings.HasPrefix(line, AttributeCommentPrefix) {
		return nil
	}
	line = strings.TrimSpace(line[len(AttributeCommentPrefix):])
	colon := strings.IndexRune(line, ':')
	var key, value string
	if colon == -1 {
		key = line
	} else {
		key, value = line[:colon], line[colon+1:]
	}
	switch key {
	case "linkage":
		return parseLinkageAttribute(value)
	case "name":
		return nameAttribute(strings.TrimSpace(value))
	case "thread_local":
		return tlsAttribute{}
	default:
		// FIXME decide what to do here. return error? log warning?
		panic("unknown attribute key: " + key)
	}
	return nil
}

type linkageAttribute llvm.Linkage

func (a linkageAttribute) Apply(v llvm.Value) {
	v.SetLinkage(llvm.Linkage(a))
}

func parseLinkageAttribute(value string) linkageAttribute {
	var result linkageAttribute
	value = strings.Replace(value, ",", " ", -1)
	for _, field := range strings.Fields(value) {
		switch strings.ToLower(field) {
		case "private":
			result |= linkageAttribute(llvm.PrivateLinkage)
		case "internal":
			result |= linkageAttribute(llvm.InternalLinkage)
		case "available_externally":
			result |= linkageAttribute(llvm.AvailableExternallyLinkage)
		case "linkonce":
			result |= linkageAttribute(llvm.LinkOnceAnyLinkage)
		case "common":
			result |= linkageAttribute(llvm.CommonLinkage)
		case "weak":
			result |= linkageAttribute(llvm.WeakAnyLinkage)
		case "appending":
			result |= linkageAttribute(llvm.AppendingLinkage)
		case "extern_weak":
			result |= linkageAttribute(llvm.ExternalWeakLinkage)
		case "linkonce_odr":
			result |= linkageAttribute(llvm.LinkOnceODRLinkage)
		case "weak_odr":
			result |= linkageAttribute(llvm.WeakODRLinkage)
		case "external":
			result |= linkageAttribute(llvm.ExternalLinkage)
		}
	}
	return result
}

type nameAttribute string

func (a nameAttribute) Apply(v llvm.Value) {
	if !v.IsAFunction().IsNil() {
		name := string(a)
		curr := v.GlobalParent().NamedFunction(name)
		if !curr.IsNil() && curr != v {
			if curr.BasicBlocksCount() != 0 {
				panic(fmt.Errorf("Want to take the name %s from a function that has a body!", name))
			}
			curr.SetName(name + "_llgo_replaced")
			curr.ReplaceAllUsesWith(llvm.ConstBitCast(v, curr.Type()))
		}
		v.SetName(name)
	} else {
		v.SetName(string(a))
	}
}

type tlsAttribute struct{}

func (tlsAttribute) Apply(v llvm.Value) {
	v.SetThreadLocal(true)
}
