import Foundation

protocol HasWhy
{
  var why : Int { get }
}

protocol ClassHasWhy : class
{
  var why : Int { get }
}

class ShouldBeWhy : NSObject, ClassHasWhy, HasWhy
{
  var before_why : Int = 0xfeedface
  var why : Int = 10
  var after_why : Int = 0xdeadbeef
}

class ClassByWhy<T> where T : ClassHasWhy 
{
  let myWhy : Int
  init(input : T)
  {
    myWhy = input.why // FIXME <rdar://problem/43057063>
  }
}

class ByWhy<T> where T : HasWhy 
{
  let myWhy : Int
  init(input : T)
  {
    myWhy = input.why //%self.expect('expr -d run -- input', substrs=['a.ShouldBeWhy', 'isa = a.ShouldBeWhy', 'before_why = 4277009102', 'why = 10'])
                      //%self.expect('expr -d run -- input', substrs=['a.ShouldBeWhy', 'isa = a.ShouldBeWhy', 'before_why = 4277009102', 'why = 10'])
  }
}

func doIt() 
{
  let mySBW = ShouldBeWhy()
  let byWhy = ByWhy(input: mySBW)
  let classByWhy = ClassByWhy(input: mySBW)
  print(byWhy.myWhy, classByWhy.myWhy)
}

doIt()
