#import <Foundation/Foundation.h>

// SourceBase will be the base class of Source.  We'll pass a Source object into a
// function as a SourceBase, and then see if the dynamic typing can get us through the KVO
// goo and all the way back to Source.

@interface SourceBase: NSObject
{
    uint32_t _value;
}
- (SourceBase *) init;
- (uint32_t) getValue;
@end

@implementation SourceBase
- (SourceBase *) init
{
    [super init];
    _value = 10;
    return self;
}
- (uint32_t) getValue
{
    return _value;
}
@end

// Source is a class that will be observed by the Observer class below.
// When Observer sets itself up to observe this property (in initWithASource)
// the KVO system will overwrite the "isa" pointer of the object with the "kvo'ed" 
// one.

@interface Source : SourceBase
{
    int _property;
}
- (Source *) init;
- (void) setProperty: (int) newValue;
@end

@implementation Source
- (Source *) init
{
    [super init];
    _property = 20;
    return self;
}
- (void) setProperty: (int) newValue
{
    _property = newValue;  // This is the line in setProperty, make sure we step to here.
}
@end

@interface SourceDerived : Source
{
    int _derivedValue;
}
- (SourceDerived *) init;
- (uint32_t) getValue;
@end

@implementation SourceDerived
- (SourceDerived *) init
{
    [super init];
    _derivedValue = 30;
    return self;
}
- (uint32_t) getValue
{
    return _derivedValue;
}
@end

// Observer is the object that will watch Source and cause KVO to swizzle it...

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

uint32_t 
handle_SourceBase (SourceBase *object)
{
    return [object getValue];  // Break here to check dynamic values.
}

int main ()
{
    Source *mySource;
    Observer *myObserver;

    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

    mySource = [[SourceDerived alloc] init];
    myObserver = [Observer observerWithSource: mySource];

    [mySource setProperty: 5];      // Break here to see if we can step into real method.
    
    uint32_t return_value = handle_SourceBase (mySource);

    SourceDerived *unwatchedSource = [[SourceDerived alloc] init];
    
    return_value = handle_SourceBase (unwatchedSource);
    
    [pool release];
    return 0;

}
