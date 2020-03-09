#import <Foundation/Foundation.h>
#include <unistd.h>
#import "my-base.h"

@interface MyString : MyBase {
    NSString *str;
    NSDate *date;
    BOOL _desc_pauses;
}

@property(retain) NSString * str_property;
@property BOOL descriptionPauses;

- (id)initWithNSString:(NSString *)string;
@end

@implementation MyString
@synthesize descriptionPauses = _desc_pauses;
@synthesize str_property = str;

- (id)initWithNSString:(NSString *)string
{
    if (self = [super init])
    {
        str = [NSString stringWithString:string];
        date = [NSDate date];
    }
    self.descriptionPauses = NO;
    return self;
}

- (void)dealloc
{
    [date release];
    [str release];
    [super dealloc];
}

- (NSString *)description
{
    // Set a breakpoint on '-[MyString description]' and test expressions:
    // expression (char *)sel_getName(_cmd)
  if (self.descriptionPauses)  // Break here for description test
    {
        printf ("\nAbout to sleep.\n");
        usleep(100000);
    }

    return [str stringByAppendingFormat:@" with timestamp: %@", date];
}
@end

int
Test_Selector ()
{
    SEL sel = @selector(length);
    printf("sel = %p\n", sel);
    // Expressions to test here for selector: 
    // expression (char *)sel_getName(sel)
    //      The expression above should return "sel" as it should be just
    //      a uniqued C string pointer. We were seeing the result pointer being
    //      truncated with recent LLDBs.
    return 0; // Break here for selector: tests
}

int
Test_NSString (const char *program)
{
    NSString *str = [NSString stringWithFormat:@"Hello from '%s'", program];
    NSLog(@"NSString instance: %@", str);
    printf("str = '%s'\n", [str cStringUsingEncoding: [NSString defaultCStringEncoding]]);
    printf("[str length] = %zu\n", (size_t)[str length]);
    printf("[str description] = %s\n", [[str description] UTF8String]);
    id str_id = str;
    // Expressions to test here for NSString:
    // expression (char *)sel_getName(sel)
    // expression [str length]
    // expression [str_id length]
    // expression [str description]
    // expression [str_id description]
    // expression str.length
    // expression str.description
    // expression str = @"new"
    // expression str = [NSString stringWithFormat: @"%cew", 'N']
    return 0; // Break here for NSString tests
}

NSString *my_global_str = NULL;

void
Test_MyString (const char *program)
{
    my_global_str = @"This is a global string";
    NSString *str = [NSString stringWithFormat:@"Hello from '%s'", program];
    MyString *my = [[MyString alloc] initWithNSString:str];
    NSLog(@"MyString instance: %@", [my description]);
    my.descriptionPauses = YES;     // Set break point at this line.  Test 'expression -o -- my'.
    NSLog(@"MyString instance: %@", [my description]);
}

int
Test_NSArray ()
{
    NSMutableArray *nil_mutable_array = nil;
    NSArray *array1 = [NSArray arrayWithObjects: @"array1 object1", @"array1 object2", @"array1 object3", nil];
    NSArray *array2 = [NSArray arrayWithObjects: array1, @"array2 object2", @"array2 object3", nil];
    // Expressions to test here for NSArray:
    // expression [nil_mutable_array count]
    // expression [array1 count]
    // expression array1.count
    // expression [array2 count]
    // expression array2.count
    id obj;
    // After each object at index call, use expression and validate object
    obj = [array1 objectAtIndex: 0]; // Break here for NSArray tests
    obj = [array1 objectAtIndex: 1];
    obj = [array1 objectAtIndex: 2];

    obj = [array2 objectAtIndex: 0];
    obj = [array2 objectAtIndex: 1];
    obj = [array2 objectAtIndex: 2];
    NSUInteger count = [nil_mutable_array count];
    return 0;
}


int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    Test_Selector();
    Test_NSArray ();
    Test_NSString (argv[0]);
    Test_MyString (argv[0]);

    printf("sizeof(id) = %zu\n", sizeof(id));
    printf("sizeof(Class) = %zu\n", sizeof(Class));
    printf("sizeof(SEL) = %zu\n", sizeof(SEL));

    [pool release];
    return 0;
}
