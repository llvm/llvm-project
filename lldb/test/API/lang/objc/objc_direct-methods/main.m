#import <Foundation/Foundation.h>

int side_effect = 0;

NSString *str = @"some string";

const char *directCallConflictingName() {
  return "wrong function";
}

@interface Foo : NSObject {
  int instance_var;
}
-(int) entryPoint;
@end

@implementation Foo
-(int) entryPoint
{
  // Try calling directly with self. Same as in the main method otherwise.
  return 0; //%self.expect("expr [self directCallNoArgs]", substrs=["called directCallNoArgs"])
            //%self.expect("expr [self directCallArgs: 1111]", substrs=["= 2345"])
            //%self.expect("expr side_effect = 0; [self directCallVoidReturn]; side_effect", substrs=["= 4321"])
            //%self.expect("expr [self directCallNSStringArg: str]", substrs=['@"some string"'])
            //%self.expect("expr [self directCallIdArg: (id)str]", substrs=['@"some string appendix"'])
            //%self.expect("expr [self directCallConflictingName]", substrs=["correct function"])
            //%self.expect("expr [self directCallWithCategory]", substrs=["called function with category"])
}

// Declare several objc_direct functions we can test.
-(const char *) directCallNoArgs __attribute__((objc_direct))
{
  return "called directCallNoArgs";
}

-(void) directCallVoidReturn __attribute__((objc_direct))
{
  side_effect = 4321;
}

-(int) directCallArgs:(int)i __attribute__((objc_direct))
{
  // Use the arg in some way to make sure that gets passed correctly.
  return i + 1234;
}

-(NSString *) directCallNSStringArg:(NSString *)str __attribute__((objc_direct))
{
  return str;
}

-(NSString *) directCallIdArg:(id)param __attribute__((objc_direct))
{
  return [param stringByAppendingString:@" appendix"];
}

// We have another function with the same name above. Make sure this doesn't influence
// what we call.
-(const char *) directCallConflictingName  __attribute__((objc_direct))
{
  return "correct function";
}
@end


@interface Foo (Cat)
@end

@implementation Foo (Cat)
-(const char *) directCallWithCategory  __attribute__((objc_direct))
{
  return "called function with category";
}
@end

int main()
{
  Foo *foo = [[Foo alloc] init];
  [foo directCallNoArgs];
  [foo directCallArgs: 1];
  [foo directCallVoidReturn];
  [foo directCallNSStringArg: str];
  [foo directCallIdArg: (id)str];
  [foo entryPoint];  //%self.expect("expr [foo directCallNoArgs]", substrs=["called directCallNoArgs"])
                     //%self.expect("expr [foo directCallArgs: 1111]", substrs=["= 2345"])
                     //%self.expect("expr side_effect = 0; [foo directCallVoidReturn]; side_effect", substrs=["= 4321"])
                     //%self.expect("expr [foo directCallNSStringArg: str]", substrs=['@"some string"'])
                     //%self.expect("expr [foo directCallIdArg: (id)str]", substrs=['@"some string appendix"'])
                     //%self.expect("expr [foo directCallConflictingName]", substrs=["correct function"])
                     //%self.expect("expr [foo directCallWithCategory]", substrs=["called function with category"])
  return 0;
}
