import Enum 

func test()
{
    let ascending = getReturn(-1)
    let descending = getReturn(1)
    let same = getReturn(0)
    return //%self.expect('frame var -d run-target -- ascending', substrs=['OrderedAscending'])
           //%self.expect('frame var -d run-target -- descending', substrs=['OrderedDescending'])
           //%self.expect('frame var -d run-target -- same', substrs=['OrderedSame'])
           //%self.expect('expr -d run-target -- ascending', substrs=['OrderedAscending'])
           //%self.expect('expr -d run-target -- descending', substrs=['OrderedDescending'])
           //%self.expect('expr -d run-target -- same', substrs=['OrderedSame'])
}

_ = test()
print("this is needed to load the stdlib")
