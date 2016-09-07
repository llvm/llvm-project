//===-- fmts_tester.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Cocoa/Cocoa.h>
#include <list>
#include <map>
#include <string>
#include <vector>

int main() {
  NSArray *nsarray = @[ @1, @2, @"hello world", @3, @4, @"foobar" ];
  NSMutableArray *nsmutablearray = [[NSMutableArray alloc] initWithCapacity:5];
  [nsmutablearray addObject:@1];
  [nsmutablearray addObject:@2];
  [nsmutablearray addObject:@"hello world"];
  [nsmutablearray addObject:@3];
  [nsmutablearray addObject:@4];
  [nsmutablearray addObject:@"foobar"];
  NSDictionary *nsdictionary =
      @{ @1 : @1,
         @2 : @2,
         @"hello" : @"world",
         @3 : @3 };
  NSMutableDictionary *nsmutabledictionary =
      [[NSMutableDictionary alloc] initWithCapacity:5];
  [nsmutabledictionary setObject:@1 forKey:@1];
  [nsmutabledictionary setObject:@2 forKey:@2];
  [nsmutabledictionary setObject:@"hello" forKey:@"world"];
  [nsmutabledictionary setObject:@3 forKey:@3];
  NSString *str0 = @"Hello world";
  NSString *str1 = @"Hello ℥";
  NSString *str2 = @"Hello world";
  NSString *str3 = @"Hello ℥";
  NSString *str4 = @"Hello world";
  NSDate *me = [NSDate dateWithNaturalLanguageString:@"April 10, 1985"];
  NSDate *cutie = [NSDate dateWithNaturalLanguageString:@"January 29, 1983"];
  NSDate *mom = [NSDate dateWithNaturalLanguageString:@"May 24, 1959"];
  NSDate *dad = [NSDate dateWithNaturalLanguageString:@"October 29, 1954"];
  NSDate *today = [NSDate dateWithNaturalLanguageString:@"March 14, 2013"];
  NSArray *bundles = [NSBundle allBundles];
  NSArray *frameworks = [NSBundle allFrameworks];
  NSSet *nsset = [NSSet setWithArray:nsarray];
  NSMutableSet *nsmutableset = [NSMutableSet setWithCapacity:5];
  [nsmutableset addObject:@1];
  [nsmutableset addObject:@2];
  [nsmutableset addObject:@"hello world"];
  [nsmutableset addObject:@3];
  [nsmutableset addObject:@4];
  [nsmutableset addObject:@"foobar"];
  std::vector<int> vector;
  vector.push_back(1);
  vector.push_back(2);
  vector.push_back(3);
  vector.push_back(4);
  vector.push_back(5);
  std::list<int> list;
  list.push_back(1);
  list.push_back(2);
  list.push_back(3);
  list.push_back(4);
  list.push_back(5);
  std::map<int, int> map;
  map[1] = 1;
  map[2] = 2;
  map[3] = 3;
  map[4] = 4;
  map[5] = 5;
  std::string sstr0("Hello world");
  std::string sstr1("Hello world");
  std::string sstr2("Hello world");
  std::string sstr3("Hello world");
  std::string sstr4("Hello world");
  int x = 0;
  for (;;)
    x++;
}