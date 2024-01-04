#include <chrono>
#include <iostream>

int main() {
  // break here
  std::chrono::nanoseconds ns{1};
  std::chrono::microseconds us{12};
  std::chrono::milliseconds ms{123};
  std::chrono::seconds s{1234};
  std::chrono::minutes min{12345};
  std::chrono::hours h{123456};

  std::chrono::days d{654321};
  std::chrono::weeks w{54321};
  std::chrono::months m{4321};
  std::chrono::years y{321};

  std::chrono::day d_0{0};
  std::chrono::day d_1{1};
  std::chrono::day d_31{31};
  std::chrono::day d_255{255};

  std::chrono::month jan = std::chrono::January;
  std::chrono::month feb = std::chrono::February;
  std::chrono::month mar = std::chrono::March;
  std::chrono::month apr = std::chrono::April;
  std::chrono::month may = std::chrono::May;
  std::chrono::month jun = std::chrono::June;
  std::chrono::month jul = std::chrono::July;
  std::chrono::month aug = std::chrono::August;
  std::chrono::month sep = std::chrono::September;
  std::chrono::month oct = std::chrono::October;
  std::chrono::month nov = std::chrono::November;
  std::chrono::month dec = std::chrono::December;

  std::chrono::month month_0{0};
  std::chrono::month month_1{1};
  std::chrono::month month_2{2};
  std::chrono::month month_3{3};
  std::chrono::month month_4{4};
  std::chrono::month month_5{5};
  std::chrono::month month_6{6};
  std::chrono::month month_7{7};
  std::chrono::month month_8{8};
  std::chrono::month month_9{9};
  std::chrono::month month_10{10};
  std::chrono::month month_11{11};
  std::chrono::month month_12{12};
  std::chrono::month month_13{13};
  std::chrono::month month_255{255};

  std::chrono::year y_min{std::chrono::year::min()};
  std::chrono::year y_0{0};
  std::chrono::year y_1970{1970};
  std::chrono::year y_2038{2038};
  std::chrono::year y_max{std::chrono::year::max()};

  std::chrono::month_day md_new_years_eve{std::chrono::December / 31};
  std::chrono::month_day md_new_year{std::chrono::January / 1};
  std::chrono::month_day md_invalid{std::chrono::month{255} / 255};

  std::chrono::month_day_last mdl_jan{std::chrono::January};
  std::chrono::month_day_last mdl_new_years_eve{std::chrono::December};

  std::chrono::year_month_day ymd_bc{std::chrono::year{-1}, std::chrono::March,
                                     std::chrono::day{255}};
  std::chrono::year_month_day ymd_year_zero{
      std::chrono::year{0}, std::chrono::month{255}, std::chrono::day{25}};
  std::chrono::year_month_day ymd_unix_epoch{
      std::chrono::year{1970}, std::chrono::January, std::chrono::day{1}};

  std::cout << "break here\n";
}
