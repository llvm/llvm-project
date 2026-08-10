#import <Foundation/Foundation.h>

@protocol MyProtocol

-(const char *)hello;

@end

static int _class_int = 123;

@interface BaseClass : NSObject
{
  int _backedInt;
  int _access_count;
}

- (int) nonexistantInt;
- (void) setNonexistantInt: (int) in_int;

- (int) myGetUnbackedInt;
- (void) mySetUnbackedInt: (int) in_int;

- (int) getAccessCount;

+ (int) propConflict;

+(BaseClass *) baseClassWithBackedInt: (int) inInt andUnbackedInt: (int) inOtherInt;

@property(getter=myGetUnbackedInt,setter=mySetUnbackedInt:) int unbackedInt;
@property int backedInt;
@property (nonatomic, assign) id <MyProtocol> idWithProtocol;
@property(class) int classInt;
@property(getter=propConflict,readonly) int propConflict;
@property(readonly,class) int propConflict;
@end

@implementation BaseClass
@synthesize unbackedInt;
@synthesize backedInt = _backedInt;

+ (BaseClass *) baseClassWithBackedInt: (int) inInt andUnbackedInt: (int) inOtherInt
{
  BaseClass *new = [[BaseClass alloc] init];
  
  new->_backedInt = inInt;
  new->unbackedInt = inOtherInt;

  return new;
}

- (int) myGetUnbackedInt
{
  // NSLog (@"Getting BaseClass::unbackedInt - %d.\n", unbackedInt);
  _access_count++;
  return unbackedInt;
}

- (void) mySetUnbackedInt: (int) in_int
{
  // NSLog (@"Setting BaseClass::unbackedInt from %d to %d.", unbackedInt, in_int);
  _access_count++;
  unbackedInt = in_int;
}

- (int) nonexistantInt
{
  // NSLog (@"Getting BaseClass::nonexistantInt - %d.\n", 5);
  _access_count++;
  return 6;
}

- (void) setNonexistantInt: (int) in_int
{
  // NSLog (@"Setting BaseClass::nonexistantInt from 7 to %d.", in_int);
  _access_count++;
}

+ (int) classInt
{
    return _class_int;
}

+ (void) setClassInt:(int) n
{
    _class_int = n;
}

- (int) getAccessCount
{
  return _access_count;
}

- (int) propConflict
{
  return 4;
}
+ (int) propConflict
{
  return 6;
}
@end

typedef BaseClass TypedefBaseClass;

int
main ()
{
  BaseClass *mine = [BaseClass baseClassWithBackedInt: 10 andUnbackedInt: 20];
  TypedefBaseClass *typedefd = mine;
  int propConflict = mine.propConflict + BaseClass.propConflict;
  
  // Set a breakpoint here.
  int nonexistant = mine.nonexistantInt;

  int backedInt = mine.backedInt;

  int unbackedInt = mine.unbackedInt;

  id idWithProtocol = mine.idWithProtocol;

  NSLog (@"Results for %p: nonexistant: %d backed: %d unbacked: %d accessCount: %d.",
         mine,
         nonexistant,
         backedInt,
         unbackedInt,
         [mine getAccessCount]);
  return 0;

}

