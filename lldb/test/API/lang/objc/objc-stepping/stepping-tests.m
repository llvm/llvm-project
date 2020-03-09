#import <Foundation/Foundation.h>
#include <stdio.h>

struct return_me
{
  int first;
  int second;
};

@interface SourceBase: NSObject
{
  struct return_me my_return;
}
- (SourceBase *) initWithFirst: (int) first andSecond: (int) second;
- (void) randomMethod;
- (struct return_me) returnsStruct;
@end

@implementation SourceBase
- (void) randomMethod
{
  printf ("Called in SourceBase version of randomMethod.\n"); // SourceBase randomMethod start line.
}

- (struct return_me) returnsStruct
{
  return my_return; // SourceBase returnsStruct start line.
}

- (SourceBase *) initWithFirst: (int) first andSecond: (int) second
{
  my_return.first = first;
  my_return.second = second;

  return self;
}
@end

@interface Source : SourceBase
{
  int _property;
}
- (void) setProperty: (int) newValue;
- (void) randomMethod;
- (struct return_me) returnsStruct;
@end

@implementation Source
- (void) setProperty: (int) newValue
{
  _property = newValue;
}

- (void) randomMethod
{
  [super randomMethod];  // Source randomMethod start line.
    printf ("Called in Source version of random method.");
}

- (struct return_me) returnsStruct
{
  printf ("Called in Source version of returnsStruct.\n");  // Source returnsStruct start line.
  return [super returnsStruct];                             // Source returnsStruct call line.  
}

@end

@interface Observer : NSObject
{
  Source *_source;
}
+ (Observer *) observerWithSource: (Source *) source;
- (Observer *) initWithASource: (Source *) source;
- (void) observeValueForKeyPath: (NSString *) path 
		       ofObject: (id) object
			 change: (NSDictionary *) change
			context: (void *) context;
@end

@implementation Observer

+ (Observer *) observerWithSource: (Source *) inSource;
{
  Observer *retval;

  retval = [[Observer alloc] initWithASource: inSource];
  return retval;
}

- (Observer *) initWithASource: (Source *) source
{
  [super init];
  _source = source;
  [_source addObserver: self 
	    forKeyPath: @"property" 
	    options: (NSKeyValueObservingOptionNew | NSKeyValueObservingOptionOld)
	    context: NULL];
  return self;
}

- (void) observeValueForKeyPath: (NSString *) path 
		       ofObject: (id) object
			 change: (NSDictionary *) change
			context: (void *) context
{
  printf ("Observer function called.\n");
  return;
}
@end

int main ()
{
  Source *mySource;
  Observer *myObserver;
  struct return_me ret_val;

  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

  mySource = [[Source alloc] init];

  [mySource randomMethod];               // Set first breakpoint here.
  ret_val = [mySource returnsStruct];    // Set second breakpoint here.

  myObserver = [Observer observerWithSource: mySource];  

  [mySource randomMethod];              // Set third breakpoint here.
  ret_val = [mySource returnsStruct];   // Set fourth breakpoint here.
  [mySource setProperty: 5];            // Set fifth breakpoint here.

  // We also had a bug where stepping into a method dispatch to nil turned
  // into continue.  So make sure that works here:

  mySource = nil;
  [mySource randomMethod];             // Set nil step breakpoint here.
  [pool release];                      // Step over nil should stop here.
  return 0;

}
