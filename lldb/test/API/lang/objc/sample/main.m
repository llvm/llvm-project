#import <Foundation/Foundation.h>


@interface MyString : NSObject {
    NSString *_string;
    NSDate *_date;
}
- (id)initWithNSString:(NSString *)string;

@property (copy) NSString *string;
@property (readonly,getter=getTheDate) NSDate *date;

- (NSDate *) getTheDate;
@end

@implementation MyString

@synthesize string = _string;
@synthesize date = _date;

- (id)initWithNSString:(NSString *)string
{
    if (self = [super init])
    {
        _string = [NSString stringWithString:string];
        _date = [NSDate date];            
    }
    return self;
}

- (void) dealloc
{
    [_date release];
    [_string release];
    [super dealloc];
}

- (NSDate *) getTheDate
{
    return _date;
}

- (NSString *)description
{
    return [_string stringByAppendingFormat:@" with timestamp: %@", _date];
}
@end

int main (int argc, char const *argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    static NSString *g_global_nsstr = @"Howdy";
    
    MyString *myStr = [[MyString alloc] initWithNSString: [NSString stringWithFormat:@"string %i", 1]];
    NSString *str1 = myStr.string;
    NSString *str2 = [NSString stringWithFormat:@"string %i", 2];
    NSString *str3 = [NSString stringWithFormat:@"string %i", 3];
    NSArray *array = [NSArray arrayWithObjects: str1, str2, str3, nil];
    NSDictionary *dict = [NSDictionary dictionaryWithObjectsAndKeys:
                            str1, @"1", 
                            str2, @"2", 
                            str3, @"3", 
                            myStr.date, @"date",
                            nil];

    id str_id = str1;
    SEL sel = @selector(length);
    [pool release];
    return 0;
}
