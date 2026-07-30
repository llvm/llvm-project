#import <Foundation/Foundation.h>

int main()
{
    @autoreleasepool
    {
        // NSArrays
        NSArray *immutable_array = @[ @"foo", @"bar" ];
        NSMutableArray *mutable_array = [NSMutableArray arrayWithCapacity:2];
        [mutable_array addObjectsFromArray:immutable_array];
        
        // NSDictionaries
        NSDictionary *immutable_dictionary = @{ @"key" : @"value" };
        NSMutableDictionary *mutable_dictionary = [NSMutableDictionary dictionaryWithCapacity:1];
        [mutable_dictionary addEntriesFromDictionary:immutable_dictionary];

        NSNumber *one = @1;

        NSLog(@"Stop here"); // Set breakpoint 0 here.
    }
}
