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

  // sys_seconds aliasses
  std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>
      ss_tp{std::chrono::seconds{0}};
  std::chrono::time_point<std::chrono::system_clock,
                          std::chrono::duration<long long>>
      ss_tp_d{std::chrono::seconds{0}};
  std::chrono::time_point<std::chrono::system_clock,
                          std::chrono::duration<long long, std::ratio<1>>>
      ss_tp_d_r{std::chrono::seconds{0}};
  std::chrono::time_point<std::chrono::system_clock,
                          std::chrono::duration<long long, std::ratio<1>>>
      ss_tp_d_r2{std::chrono::seconds{0}};

  // sys_seconds
  std::chrono::sys_seconds ss_0{std::chrono::seconds{0}};
  std::chrono::sys_seconds ss_neg_date_time{
      std::chrono::seconds{-1'096'193'779'200}};
  std::chrono::sys_seconds ss_neg_seconds{
      std::chrono::seconds{-1'096'193'779'201}};
  std::chrono::sys_seconds ss_pos_date_time{
      std::chrono::seconds{971'890'963'199}};
  std::chrono::sys_seconds ss_pos_seconds{
      std::chrono::seconds{971'890'963'200}};
  std::chrono::sys_seconds ss_min{
      std::chrono::seconds{std::numeric_limits<long long>::min()}};
  std::chrono::sys_seconds ss_max{
      std::chrono::seconds{std::numeric_limits<long long>::max()}};

  // sys_days aliasses
  std::chrono::time_point<std::chrono::system_clock, std::chrono::days> sd_tp{
      std::chrono::days{0}};
  std::chrono::time_point<std::chrono::system_clock,
                          std::chrono::duration<int, std::ratio<86400>>>
      sd_tp_d_r{std::chrono::days{0}};
  std::chrono::time_point<std::chrono::system_clock,
                          std::chrono::duration<int, std::ratio<86400, 1>>>
      sd_tp_d_r2{std::chrono::days{0}};

  // sys_days
  std::chrono::sys_days sd_0{std::chrono::days{0}};

  std::chrono::sys_days sd_neg_date{std::chrono::days{-12'687'428}};
  std::chrono::sys_days sd_neg_days{std::chrono::days{-12'687'429}};

  std::chrono::sys_days sd_pos_date{std::chrono::days{11'248'737}};
  std::chrono::sys_days sd_pos_days{std::chrono::days{11'248'738}};

  std::chrono::sys_days sd_min{
      std::chrono::days{std::numeric_limits<int>::min()}};
  std::chrono::sys_days sd_max{
      std::chrono::days{std::numeric_limits<int>::max()}};

  // local_seconds aliasses
  std::chrono::time_point<std::chrono::local_t, std::chrono::seconds> ls_tp{
      std::chrono::seconds{0}};
  std::chrono::time_point<std::chrono::local_t,
                          std::chrono::duration<long long>>
      ls_tp_d{std::chrono::seconds{0}};
  std::chrono::time_point<std::chrono::local_t,
                          std::chrono::duration<long long, std::ratio<1>>>
      ls_tp_d_r{std::chrono::seconds{0}};
  std::chrono::time_point<std::chrono::local_t,
                          std::chrono::duration<long long, std::ratio<1>>>
      ls_tp_d_r2{std::chrono::seconds{0}};

  // local_seconds
  std::chrono::local_seconds ls_0{std::chrono::seconds{0}};
  std::chrono::local_seconds ls_neg_date_time{
      std::chrono::seconds{-1'096'193'779'200}};
  std::chrono::local_seconds ls_neg_seconds{
      std::chrono::seconds{-1'096'193'779'201}};
  std::chrono::local_seconds ls_pos_date_time{
      std::chrono::seconds{971'890'963'199}};
  std::chrono::local_seconds ls_pos_seconds{
      std::chrono::seconds{971'890'963'200}};
  std::chrono::local_seconds ls_min{
      std::chrono::seconds{std::numeric_limits<long long>::min()}};
  std::chrono::local_seconds ls_max{
      std::chrono::seconds{std::numeric_limits<long long>::max()}};

  // local_days aliasses
  std::chrono::time_point<std::chrono::local_t, std::chrono::days> ld_tp{
      std::chrono::days{0}};
  std::chrono::time_point<std::chrono::local_t,
                          std::chrono::duration<int, std::ratio<86400>>>
      ld_tp_d_r{std::chrono::days{0}};
  std::chrono::time_point<std::chrono::local_t,
                          std::chrono::duration<int, std::ratio<86400, 1>>>
      ld_tp_d_r2{std::chrono::days{0}};

  // local_days
  std::chrono::local_days ld_0{std::chrono::days{0}};

  std::chrono::local_days ld_neg_date{std::chrono::days{-12'687'428}};
  std::chrono::local_days ld_neg_days{std::chrono::days{-12'687'429}};

  std::chrono::local_days ld_pos_date{std::chrono::days{11'248'737}};
  std::chrono::local_days ld_pos_days{std::chrono::days{11'248'738}};

  std::chrono::local_days ld_min{
      std::chrono::days{std::numeric_limits<int>::min()}};
  std::chrono::local_days ld_max{
      std::chrono::days{std::numeric_limits<int>::max()}};

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

  std::chrono::weekday sun = std::chrono::Sunday;
  std::chrono::weekday mon = std::chrono::Monday;
  std::chrono::weekday tue = std::chrono::Tuesday;
  std::chrono::weekday wed = std::chrono::Wednesday;
  std::chrono::weekday thu = std::chrono::Thursday;
  std::chrono::weekday fri = std::chrono::Friday;
  std::chrono::weekday sat = std::chrono::Saturday;

  std::chrono::weekday weekday_0{0};
  std::chrono::weekday weekday_1{1};
  std::chrono::weekday weekday_2{2};
  std::chrono::weekday weekday_3{3};
  std::chrono::weekday weekday_4{4};
  std::chrono::weekday weekday_5{5};
  std::chrono::weekday weekday_6{6};
  std::chrono::weekday weekday_7{7};
  std::chrono::weekday weekday_8{8};
  std::chrono::weekday weekday_255{255};

  std::chrono::weekday_indexed wdi_saturday_0{std::chrono::Saturday, 0};
  std::chrono::weekday_indexed wdi_monday_1{std::chrono::Monday, 1};
  std::chrono::weekday_indexed wdi_invalid{std::chrono::weekday{255}, 255};

  std::chrono::weekday_last wdl_monday{std::chrono::Monday};
  std::chrono::weekday_last wdl_invalid{std::chrono::weekday{255}};

  std::chrono::month_day md_new_years_eve{std::chrono::December / 31};
  std::chrono::month_day md_new_year{std::chrono::January / 1};
  std::chrono::month_day md_invalid{std::chrono::month{255} / 255};

  std::chrono::month_day_last mdl_jan{std::chrono::January};
  std::chrono::month_day_last mdl_new_years_eve{std::chrono::December};

  std::chrono::month_weekday mwd_first_thursday{
      std::chrono::January,
      std::chrono::weekday_indexed{std::chrono::Thursday, 1}};

  std::chrono::month_weekday_last mwdl_last_saturday{
      std::chrono::December, std::chrono::weekday_last{std::chrono::Saturday}};

  std::chrono::year_month ym_year_zero{std::chrono::year{0},
                                       std::chrono::January};

  std::chrono::year_month_day ymd_bc{std::chrono::year{-1}, std::chrono::March,
                                     std::chrono::day{255}};
  std::chrono::year_month_day ymd_year_zero{
      std::chrono::year{0}, std::chrono::month{255}, std::chrono::day{25}};
  std::chrono::year_month_day ymd_unix_epoch{
      std::chrono::year{1970}, std::chrono::January, std::chrono::day{1}};

  std::chrono::year_month_day_last ymdl_bc{
      std::chrono::year{-1},
      std::chrono::month_day_last{std::chrono::December}};
  std::chrono::year_month_day_last ymdl_may_1970{
      std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::May}};

  std::chrono::year_month_weekday ymwd_bc{
      std::chrono::year{-1}, std::chrono::June,
      std::chrono::weekday_indexed{std::chrono::Wednesday, 2}};
  std::chrono::year_month_weekday ymwd_forth_tuesday_2024{
      std::chrono::year{2024}, std::chrono::January,
      std::chrono::weekday_indexed{std::chrono::Tuesday, 4}};

  std::chrono::year_month_weekday_last ymwdl_bc{
      std::chrono::year{-1}, std::chrono::April,
      std::chrono::weekday_last{std::chrono::Friday}};
  std::chrono::year_month_weekday_last ymwdl_2024_last_tuesday_january{
      std::chrono::year{2024}, std::chrono::January,
      std::chrono::weekday_last{std::chrono::Tuesday}};

  std::cout << "break here\n";
}
