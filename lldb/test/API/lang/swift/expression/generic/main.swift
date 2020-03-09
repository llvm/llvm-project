// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------


protocol HasSomething
{
    var something: Int
    {
        get
    }    
}

struct S<T> : HasSomething
{
    var something : Int
    {
        get { return m_something }
    }

    let m_s : T
    let m_something : Int
    init (in_t : T)
    {
        m_s = in_t
        m_something = 5
    }
}

func a (_ i : Int)
{
  var x = i // breakpoint 1
}

// Class, function

class B
{
  var m_t : Int
  var m_s : S<Int>

  init ()
  {
      m_t = 2
      m_s = S<Int>(in_t: 2)
  }

  func b (_ i : Int)
  {
    var x = i // breakpoint 2
    m_t = i
  }
}

// Generic class, function

class C<T>
{
  var m_t : T
  var m_s : S<T>

  init (input : T)
  {
    m_t = input
    m_s = S<T>(in_t: input)
  }

  func c (_ i : T)
  {
    var x = i // breakpoint 3
    m_t = i
  }
}

// Generic function

func d<U> (_ i : U)
{
  var x = i // breakpoint 4
}

// Class, generic function

class E
{
  var m_t : Int
  var m_s : S<Int>

  init ()
  {
    m_t = 5
    m_s = S<Int>(in_t : 5)
  }

  func e<U> (_ i : U)
  {
    var x = i // breakpoint 5
    m_t = 6
  }
}

// Generic class, generic function

class F<T>
{
  var m_t : T
  var m_s : S<T>

  init (input: T)
  {
    m_t = input
    m_s = S<T>(in_t: input)
  }

  func f<U> (_ i : T, _ j : U)
  {
    var x = i
    var y = j // breakpoint 6
    m_t = i
  }
}

// Invocations

a(1)

var myB = B()
myB.b(2)

var myC = C<Int>(input: 3)
myC.c(3)

d(4)

var myE = E()
myE.e(5)

var myF = F<Int>(input : 6)
myF.f(6, 7)
