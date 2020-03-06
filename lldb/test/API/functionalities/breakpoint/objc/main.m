#import <Foundation/Foundation.h>
#include <unistd.h>

@interface MyClass : NSObject
@end

@implementation MyClass : NSObject
@end

@implementation MyClass (MyCategory)


- (void) myCategoryFunction {
    NSLog (@"myCategoryFunction");
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
    Test_Selector(); // Set breakpoint here
    Test_NSArray ();
    Test_NSString (argv[0]);
    MyClass *my_class = [[MyClass alloc] init];
    [my_class myCategoryFunction];
    printf("sizeof(id) = %zu\n", sizeof(id));
    printf("sizeof(Class) = %zu\n", sizeof(Class));
    printf("sizeof(SEL) = %zu\n", sizeof(SEL));

    [pool release];
    return 0;
}
